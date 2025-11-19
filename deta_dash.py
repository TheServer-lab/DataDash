# deta_dash.py
import re
import io
import sys
import copy
import ast
import base64
import uuid
from datetime import datetime, timezone
from typing import Any, Tuple

__all__ = ["loads", "dumps", "load", "dump", "DetaDashError"]

# -------------------------
# Errors
# -------------------------
class DetaDashError(Exception):
    pass

class DetaDashParseError(DetaDashError):
    def __init__(self, message, line=None, col=None):
        loc = f" at line {line}, col {col}" if line is not None and col is not None else ""
        super().__init__(f"{message}{loc}")
        self.line = line
        self.col = col

# -------------------------
# Tokenizer
# -------------------------
TOKEN_REGEX = re.compile(
    r"""
    (?P<WHITESPACE>\s+)
  | (?P<COMMENT>//[^\n]*)
  | (?P<COMMENT2>\#[^\n]*)
  | (?P<TRIPLE_DQ>""" + r'"""' + r""")
  | (?P<TRIPLE_SQ>''' + "'''"+ r""")
  | (?P<DQ>"(?:\\.|[^"\\])*")
  | (?P<SQ>'(?:\\.|[^'\\])*')
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<LBRACK>\[)
  | (?P<RBRACK>\])
  | (?P<COLON>:)
  | (?P<COMMA>,)
  | (?P<AT>@)
  | (?P<BANG>!)
  | (?P<EQUALS>=)
  | (?P<ASTERISK>\*)
  | (?P<AMP>&)
  | (?P<NUMBER>-?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+(\.\d+)?([eE][+-]?\d+)?))
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_\-]*)
  """,
    re.VERBOSE | re.DOTALL,
)

# We'll handle triple-quoted string contents specially
class Token:
    def __init__(self, type_, value, line, col):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col
    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r}, line={self.line},col={self.col})"

def tokenize(text: str):
    pos = 0
    line = 1
    col = 1
    L = len(text)
    while pos < L:
        m = TOKEN_REGEX.match(text, pos)
        if not m:
            # Could be triple-string content starting right after triple quote token; handle error:
            # produce a helpful error
            snippet = text[pos:pos+20].replace("\n","\\n")
            raise DetaDashParseError(f"Unexpected token: {snippet!r}", line, col)
        kind = m.lastgroup
        raw = m.group(kind)
        start = pos
        pos = m.end()
        # compute new line/col
        newlines = raw.count("\n")
        if kind == "WHITESPACE" or kind.startswith("COMMENT"):
            # skip
            if newlines:
                line += newlines
                col = 1 + len(raw.rsplit("\n", 1)[-1])
            else:
                col += len(raw)
            continue
        if kind == "TRIPLE_DQ" or kind == "TRIPLE_SQ":
            # find the closing triple quote
            quote = '"""' if kind == "TRIPLE_DQ" else "'''"
            end_idx = text.find(quote, pos)
            if end_idx == -1:
                raise DetaDashParseError("Unterminated triple-quoted string", line, col)
            content = text[pos:end_idx]
            # advance pos past closing triple
            old_pos = pos
            pos = end_idx + len(quote)
            # update line/col counts based on content + closing quote
            newlines = content.count("\n")
            if newlines:
                line += newlines
                col = 1 + len(content.rsplit("\n", 1)[-1]) + len(quote)
            else:
                col += len(content) + len(quote)
            yield Token("TRI_STRING", content, line - newlines, col - len(content) - len(quote))
            continue
        # Normal tokens
        token_value = raw
        if kind in ("DQ", "SQ"):
            # strip quotes and unescape basic escapes
            inner = token_value[1:-1]
            inner = inner.encode("utf-8").decode("unicode_escape")
            token_value = inner
            kind = "STRING"
        if kind == "NUMBER":
            token_value = token_value
        yield Token(kind, token_value, line, col)
        # update line/col
        if newlines:
            line += newlines
            col = 1 + len(raw.rsplit("\n", 1)[-1])
        else:
            col += len(raw)

# -------------------------
# Safe expression evaluator (AST)
# -------------------------
ALLOWED_NODES = {
    ast.Expression,
    ast.UnaryOp,
    ast.BinOp,
    ast.Num,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.FloorDiv,
    ast.BitXor,
    ast.BitAnd,
    ast.BitOr,
    ast.LShift,
    ast.RShift,
    ast.Call,      # will disallow later
    ast.Name,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Subscript,
    ast.Index,
    ast.Slice,
}

def safe_eval_expr(expr_src: str, context: dict):
    """
    Evaluate arithmetic expression safely.
    Allowed: numeric constants, binary/unary ops, names referencing numeric values in context.
    """
    try:
        node = ast.parse(expr_src, mode="eval")
    except SyntaxError as e:
        raise DetaDashError(f"Invalid expression: {expr_src!r}: {e}")
    # Walk nodes and ensure safety
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            raise DetaDashError("Function calls not allowed in expressions")
        if isinstance(n, (ast.Attribute, ast.Import, ast.ImportFrom, ast.Lambda, ast.DictComp, ast.ListComp, ast.GeneratorExp)):
            raise DetaDashError("Disallowed expression construct")
    # transform Names into values from context
    class NameReplacer(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id not in context:
                raise DetaDashError(f"Unknown name in expression: {n.id!r}")
            v = context[n.id]
            if isinstance(v, (int, float)):
                return ast.copy_location(ast.Constant(value=v), n)
            else:
                # try to convert to number if possible
                try:
                    fv = float(v)
                    return ast.copy_location(ast.Constant(value=fv), n)
                except Exception:
                    raise DetaDashError(f"Name {n.id!r} does not resolve to a number")
    try:
        new_node = NameReplacer().visit(node)
        ast.fix_missing_locations(new_node)
        compiled = compile(new_node, "<deta_expr>", "eval")
        return eval(compiled, {"__builtins__": {}}, {})
    except DetaDashError:
        raise
    except Exception as e:
        raise DetaDashError(f"Error evaluating expression {expr_src!r}: {e}")

# -------------------------
# Parser (recursive descent)
# -------------------------
class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.i = 0
        self.anchors = {}
        self.root_context = {}  # used for expression evaluation across top-level names
        # for tracking current path to help error messages
    def peek(self) -> Token:
        if self.i < len(self.tokens):
            return self.tokens[self.i]
        return Token("EOF", "", -1, -1)
    def next(self) -> Token:
        t = self.peek()
        self.i += 1
        return t
    def expect(self, type_):
        t = self.next()
        if t.type != type_:
            raise DetaDashParseError(f"Expected token {type_!r} but got {t.type!r}", t.line, t.col)
        return t
    def accept(self, type_):
        t = self.peek()
        if t.type == type_:
            return self.next()
        return None

    def parse(self):
        # Allow root to be object or list or single value
        if self.peek().type == "LBRACE":
            obj = self.parse_object()
            # fill root context keys for expressions
            if isinstance(obj, dict):
                self.root_context.update(obj)
            return obj
        if self.peek().type == "LBRACK":
            arr = self.parse_array()
            return arr
        # Could be sequence of key: value pairs without surrounding braces - but spec requires object or list root; allow object-like top-level by reading pairs until EOF
        # We will try to parse as single object if tokens start with IDENT/STRING etc. and next is COLON
        # Peekahead
        # fallback: parse single value
        val = self.parse_value()
        return val

    def parse_object(self):
        t = self.expect("LBRACE")
        obj = {}
        while True:
            if self.accept("RBRACE"):
                break
            # Accept optional anchor before key: &name { ... } OR &name IDENT : value
            if self.peek().type == "AMP":
                # anchor applies to next value (object)
                self.next()  # consume &
                name_token = self.peek()
                if name_token.type not in ("IDENT", "STRING"):
                    raise DetaDashParseError("Invalid anchor name", name_token.line, name_token.col)
                self.next()
                name = name_token.value
                # Next token should be LBRACE indicating an anchored object, or COLON if anchored key?
                # We'll support &name { ... } and &name followed by object value at position of a value.
                # But in our grammar anchor usage in spec is: key: &name { ... } — anchor appears as value prefix.
                # Here inside object parsing we expect key: value, so handle normal key parsing below and detect anchors when parsing value.
                # Roll back: treat anchor token as unexpected here (the anchor should appear where a value is expected).
                # So we put it back and continue with key parsing.
                # For now, just raise.
                raise DetaDashParseError("Unexpected anchor here", name_token.line, name_token.col)
            # parse key
            key_token = self.next()
            if key_token.type not in ("IDENT", "STRING"):
                raise DetaDashParseError("Expected key (identifier or string) in object", key_token.line, key_token.col)
            key = key_token.value
            self.expect("COLON")
            # parse value (with anchor detection)
            value = self.parse_value(allow_anchor=True)
            obj[key] = value
            # update root_context for top-level keys (helpful for expressions referencing top-level)
            self.root_context[key] = value
            # comma or closing
            if self.accept("COMMA"):
                # allow trailing comma then possibly RBRACE
                if self.peek().type == "RBRACE":
                    continue
                else:
                    continue
            elif self.peek().type == "RBRACE":
                self.next()  # consume
                break
            else:
                # missing comma or closing brace
                t = self.peek()
                raise DetaDashParseError("Expected ',' or '}' after object item", t.line, t.col)
        return obj

    def parse_array(self):
        self.expect("LBRACK")
        arr = []
        while True:
            if self.accept("RBRACK"):
                break
            value = self.parse_value()
            arr.append(value)
            if self.accept("COMMA"):
                if self.peek().type == "RBRACK":
                    continue
                else:
                    continue
            elif self.peek().type == "RBRACK":
                self.next()
                break
            else:
                t = self.peek()
                raise DetaDashParseError("Expected ',' or ']' after array item", t.line, t.col)
        return arr

    def parse_value(self, allow_anchor=False):
        t = self.peek()
        # Anchors when used as value prefix: &name <value>
        if allow_anchor and t.type == "AMP":
            self.next()  # consume &
            name_token = self.next()
            if name_token.type not in ("IDENT", "STRING"):
                raise DetaDashParseError("Expected anchor name after '&'", name_token.line, name_token.col)
            anchor_name = name_token.value
            # next token should be LBRACE or LBRACK or plain value
            val = self.parse_value(allow_anchor=False)
            # store anchor as deep copy
            self.anchors[anchor_name] = copy.deepcopy(val)
            return val

        # Reference: *name
        if t.type == "ASTERISK":
            self.next()
            name_token = self.next()
            if name_token.type not in ("IDENT", "STRING"):
                raise DetaDashParseError("Expected name after '*'", name_token.line, name_token.col)
            name = name_token.value
            if name not in self.anchors:
                raise DetaDashParseError(f"Unknown anchor reference: {name}", name_token.line, name_token.col)
            return copy.deepcopy(self.anchors[name])

        # Expressions: '=' token then expression until comma or closing brace/bracket
        if t.type == "EQUALS":
            # consume equals
            self.next()
            # gather tokens until comma or RBRACE or RBRACK
            expr_parts = []
            while True:
                nt = self.peek()
                if nt.type in ("COMMA", "RBRACE", "RBRACK", "EOF"):
                    break
                expr_parts.append(self.next().value)
            expr_src = " ".join(str(x) for x in expr_parts).strip()
            try:
                # evaluate using root_context (top-level keys/values) merged with local variables
                val = safe_eval_expr(expr_src, self.root_context)
                return val
            except DetaDashError as e:
                raise DetaDashParseError(f"Error evaluating expression: {e}", t.line, t.col)

        # Extended tags: @ and ! (e.g., @2025-..., @uuid("..."), !base64("..."))
        if t.type == "AT":
            self.next()
            # next token may be IDENT (like uuid) or NUMBER/IDENT/STRING token immediate like date literal
            next_token = self.next()
            if next_token.type == "IDENT":
                ident = next_token.value
                # If followed by STRING or LPAREN-like syntax, handle patterns like @uuid("...")
                if self.peek().type == "LBRACE":
                    # unlikely
                    pass
                # Next token may be STRING or TRI_STRING
                if self.peek().type == "STRING":
                    param = self.next().value
                    return self.handle_at_tag(ident, param)
                else:
                    # If ident was actually an ISO date like 2025-11-..., allow that, but tokenization would not have produced IDENT for digits. So fallback:
                    raise DetaDashParseError("Unsupported @ literal form", next_token.line, next_token.col)
            elif next_token.type == "STRING":
                # @ "..." ??? treat as raw tagged string
                return next_token.value
            elif next_token.type == "NUMBER":
                # maybe date-like without quotes - pass raw
                raw = next_token.value
                return self.handle_at_date_literal(raw)
            else:
                # Perhaps date-literal tokenization didn't split; fallback to reading identifier-like characters from token stream
                raise DetaDashParseError("Unsupported @ literal token", next_token.line, next_token.col)

        if t.type == "BANG":
            # !base64("...")
            self.next()
            next_token = self.peek()
            if next_token.type == "IDENT":
                ident = self.next().value
                # Expect "(" encoded as IDENT? We simply accept next STRING token as parameter
                if self.peek().type == "STRING":
                    param = self.next().value
                    return self.handle_bang_tag(ident, param)
                else:
                    raise DetaDashParseError("Expected string parameter for !tag", next_token.line, next_token.col)
            else:
                raise DetaDashParseError("Invalid '!' tag usage", t.line, t.col)

        # Objects and arrays
        if t.type == "LBRACE":
            return self.parse_object()
        if t.type == "LBRACK":
            return self.parse_array()
        # Strings
        if t.type == "STRING" or t.type == "TRI_STRING":
            self.next()
            return t.value
        # Numbers
        if t.type == "NUMBER":
            self.next()
            s = t.value
            try:
                if s.startswith(("0x","0X")):
                    return int(s, 16)
                if s.startswith(("0b","0B")):
                    return int(s, 2)
                if "." in s or "e" in s or "E" in s:
                    return float(s)
                return int(s)
            except Exception:
                # maybe Infinity/NaN
                if s in ("Infinity", "-Infinity", "NaN"):
                    if s == "NaN":
                        return float("nan")
                    if s == "Infinity":
                        return float("inf")
                    return float("-inf")
                raise DetaDashParseError(f"Invalid number literal: {s}", t.line, t.col)
        # Identifiers: true/false/null or unquoted bare strings or numbers/dates starting w/ @
        if t.type == "IDENT":
            self.next()
            v = t.value
            if v == "true":
                return True
            if v == "false":
                return False
            if v == "null":
                return None
            # Could be date literal that starts with digits, but tokenizer won't make digits into IDENT.
            # Return as string for unknown identifiers
            return v
        if t.type == "EOF":
            return None
        # Anything else
        raise DetaDashParseError(f"Unexpected token {t.type}", t.line, t.col)

    def handle_at_tag(self, ident, param):
        # ident like uuid
        if ident.lower() == "uuid":
            try:
                return uuid.UUID(param)
            except Exception as e:
                raise DetaDashParseError(f"Invalid UUID literal: {param}: {e}")
        # other tags, fallback: treat as string with tag
        return {"@"+ident: param}

    def handle_at_date_literal(self, raw):
        # attempt ISO parse
        try:
            if raw.endswith("Z"):
                # strip Z and set tz=utc
                return datetime.fromisoformat(raw[:-1]).replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(raw)
        except Exception:
            raise DetaDashParseError(f"Invalid date literal: {raw}")

    def handle_bang_tag(self, ident, param):
        if ident.lower() == "base64":
            try:
                return base64.b64decode(param)
            except Exception as e:
                raise DetaDashParseError(f"Invalid base64 data: {e}")
        return {"!"+ident: param}

# -------------------------
# Public API: loads/dumps/load/dump
# -------------------------
def loads(s: str) -> Any:
    """Parse a DetaDash string and return Python data structure."""
    tokens = list(tokenize(s))
    p = Parser(tokens)
    val = p.parse()
    # After full parse, try to resolve expressions that may reference top-level names (already handled during parse via root_context)
    return val

def load(fp) -> Any:
    text = fp.read()
    return loads(text)

# Simple serializer to DetaDash format (not full featured, but good for config dumps)
def _serialize_value(v, indent=0):
    pad = "  " * indent
    if isinstance(v, dict):
        items = []
        for k, val in v.items():
            key = k if re.match(r"^[A-Za-z_][A-Za-z0-9_\-]*$", k) else f'"{k}"'
            items.append(f"{pad}  {key}: {_serialize_value(val, indent+1)},")
        inner = "\n".join(items)
        return "{\n" + inner + "\n" + pad + "}"
    if isinstance(v, list):
        items = [f"{pad}  {_serialize_value(x, indent+1)}," for x in v]
        inner = "\n".join(items)
        return "[\n" + inner + "\n" + pad + "]"
    if isinstance(v, str):
        # choose quoting, escape quotes
        esc = v.replace("\\", "\\\\").replace('"', '\\"')
        if "\n" in v:
            return '"""' + esc + '"""'
        return '"' + esc + '"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, bytes):
        return '!base64(' + '"' + base64.b64encode(v).decode('ascii') + '"' + ')'
    if isinstance(v, uuid.UUID):
        return '@uuid("' + str(v) + '")'
    if isinstance(v, datetime):
        if v.tzinfo is None:
            iso = v.isoformat()
        else:
            iso = v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return f"@{iso}"
    # fallback
    return '"' + str(v) + '"'

def dumps(obj: Any, pretty: bool = True) -> str:
    return _serialize_value(obj, 0)

def dump(obj: Any, fp):
    fp.write(dumps(obj))

# -------------------------
# Demo / simple tests
# -------------------------
if __name__ == "__main__":
    example = r'''
{
    // simple comment
    name: "DetaDash",
    version: "1.0",
    debug: true,
    window: {
        width: 800,
        height: 600,
    },
    colors: [
        "red",
        "green",
        "blue",
    ],
    created: @2025-11-19T10:22:00Z,
    description: """
This is a
multi-line description.
""",
    shared: &cfg {
        lives: 3,
        ammo: 90,
    },
    clone: *cfg,
    area: = window.width * window.height,
}
'''
    print("Parsing example DetaDash...")
    try:
        data = loads(example)
        from pprint import pprint
        pprint(data)
        print("\nSerialized back to DetaDash-like text:")
        print(dumps(data))
    except Exception as e:
        print("Error:", e)
