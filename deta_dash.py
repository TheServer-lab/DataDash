# deta_dash.py (fixed)
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
# Regex for tokens excluding triple-quoted strings (they are handled manually)
TOKEN_REGEX = re.compile(
    r'''
    (?P<WHITESPACE>\s+)
  | (?P<COMMENT>//[^\n]*)
  | (?P<COMMENT2>\#[^\n]*)
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
  ''',
    re.VERBOSE | re.DOTALL,
)

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
        # Handle triple-quoted strings manually first
        if text.startswith('"""', pos) or text.startswith("'''", pos):
            quote = '"""' if text.startswith('"""', pos) else "'''"
            start_line = line
            start_col = col
            # find closing
            end_idx = text.find(quote, pos + 3)
            if end_idx == -1:
                raise DetaDashParseError("Unterminated triple-quoted string", line, col)
            content = text[pos+3:end_idx]
            # update line/col
            newlines = content.count("\n")
            if newlines:
                line += newlines
                col = 1 + len(content.rsplit("\n", 1)[-1]) + 3
            else:
                col += len(content) + 3
            pos = end_idx + 3
            yield Token("TRI_STRING", content, start_line, start_col)
            continue

        m = TOKEN_REGEX.match(text, pos)
        if not m:
            snippet = text[pos:pos+40].replace("\n","\\n")
            raise DetaDashParseError(f"Unexpected token: {snippet!r}", line, col)
        kind = m.lastgroup
        raw = m.group(kind)
        start = pos
        pos = m.end()
        newlines = raw.count("\n")
        if kind == "WHITESPACE" or kind.startswith("COMMENT"):
            if newlines:
                line += newlines
                col = 1 + len(raw.rsplit("\n", 1)[-1])
            else:
                col += len(raw)
            continue
        token_value = raw
        if kind in ("DQ", "SQ"):
            inner = token_value[1:-1]
            # unescape common escapes
            try:
                inner = bytes(inner, "utf-8").decode("unicode_escape")
            except Exception:
                pass
            token_value = inner
            kind = "STRING"
        yield Token(kind, token_value, line, col)
        if newlines:
            line += newlines
            col = 1 + len(raw.rsplit("\n", 1)[-1])
        else:
            col += len(raw)

# -------------------------
# Safe expression evaluator (AST)
# -------------------------
def safe_eval_expr(expr_src: str, context: dict):
    try:
        node = ast.parse(expr_src, mode="eval")
    except SyntaxError as e:
        raise DetaDashError(f"Invalid expression: {expr_src!r}: {e}")
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            raise DetaDashError("Function calls not allowed in expressions")
        if isinstance(n, (ast.Attribute, ast.Import, ast.ImportFrom, ast.Lambda, ast.DictComp, ast.ListComp, ast.GeneratorExp)):
            raise DetaDashError("Disallowed expression construct")
    class NameReplacer(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id not in context:
                raise DetaDashError(f"Unknown name in expression: {n.id!r}")
            v = context[n.id]
            if isinstance(v, (int, float)):
                return ast.copy_location(ast.Constant(value=v), n)
            else:
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
        self.root_context = {}
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
        if self.peek().type == "LBRACE":
            obj = self.parse_object()
            if isinstance(obj, dict):
                self.root_context.update(obj)
            return obj
        if self.peek().type == "LBRACK":
            return self.parse_array()
        return self.parse_value()

    def parse_object(self):
        self.expect("LBRACE")
        obj = {}
        while True:
            if self.accept("RBRACE"):
                break
            key_token = self.next()
            if key_token.type not in ("IDENT", "STRING"):
                raise DetaDashParseError("Expected key (identifier or string) in object", key_token.line, key_token.col)
            key = key_token.value
            self.expect("COLON")
            value = self.parse_value(allow_anchor=True)
            obj[key] = value
            self.root_context[key] = value
            if self.accept("COMMA"):
                if self.peek().type == "RBRACE":
                    continue
                else:
                    continue
            elif self.peek().type == "RBRACE":
                self.next()
                break
            else:
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
        if allow_anchor and t.type == "AMP":
            self.next()
            name_token = self.next()
            if name_token.type not in ("IDENT", "STRING"):
                raise DetaDashParseError("Expected anchor name after '&'", name_token.line, name_token.col)
            anchor_name = name_token.value
            val = self.parse_value(allow_anchor=False)
            self.anchors[anchor_name] = copy.deepcopy(val)
            return val
        if t.type == "ASTERISK":
            self.next()
            name_token = self.next()
            if name_token.type not in ("IDENT", "STRING"):
                raise DetaDashParseError("Expected name after '*'", name_token.line, name_token.col)
            name = name_token.value
            if name not in self.anchors:
                raise DetaDashParseError(f"Unknown anchor reference: {name}", name_token.line, name_token.col)
            return copy.deepcopy(self.anchors[name])
        if t.type == "EQUALS":
            self.next()
            expr_parts = []
            while True:
                nt = self.peek()
                if nt.type in ("COMMA", "RBRACE", "RBRACK", "EOF"):
                    break
                expr_parts.append(self.next().value)
            expr_src = " ".join(str(x) for x in expr_parts).strip()
            try:
                val = safe_eval_expr(expr_src, self.root_context)
                return val
            except DetaDashError as e:
                raise DetaDashParseError(f"Error evaluating expression: {e}", t.line, t.col)
        if t.type == "AT":
            self.next()
            next_token = self.next()
            if next_token.type == "IDENT":
                ident = next_token.value
                if self.peek().type == "STRING":
                    param = self.next().value
                    return self.handle_at_tag(ident, param)
                else:
                    raise DetaDashParseError("Unsupported @ literal form", next_token.line, next_token.col)
            elif next_token.type == "STRING":
                return next_token.value
            elif next_token.type == "NUMBER":
                raw = next_token.value
                return self.handle_at_date_literal(raw)
            else:
                raise DetaDashParseError("Unsupported @ literal token", next_token.line, next_token.col)
        if t.type == "BANG":
            self.next()
            next_token = self.peek()
            if next_token.type == "IDENT":
                ident = self.next().value
                if self.peek().type == "STRING":
                    param = self.next().value
                    return self.handle_bang_tag(ident, param)
                else:
                    raise DetaDashParseError("Expected string parameter for !tag", next_token.line, next_token.col)
            else:
                raise DetaDashParseError("Invalid '!' tag usage", t.line, t.col)
        if t.type == "LBRACE":
            return self.parse_object()
        if t.type == "LBRACK":
            return self.parse_array()
        if t.type in ("STRING", "TRI_STRING"):
            self.next()
            return t.value
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
                if s in ("Infinity", "-Infinity", "NaN"):
                    if s == "NaN":
                        return float("nan")
                    if s == "Infinity":
                        return float("inf")
                    return float("-inf")
                raise DetaDashParseError(f"Invalid number literal: {s}", t.line, t.col)
        if t.type == "IDENT":
            self.next()
            v = t.value
            if v == "true":
                return True
            if v == "false":
                return False
            if v == "null":
                return None
            return v
        if t.type == "EOF":
            return None
        raise DetaDashParseError(f"Unexpected token {t.type}", t.line, t.col)

    def handle_at_tag(self, ident, param):
        if ident.lower() == "uuid":
            try:
                return uuid.UUID(param)
            except Exception as e:
                raise DetaDashParseError(f"Invalid UUID literal: {param}: {e}")
        return {"@"+ident: param}

    def handle_at_date_literal(self, raw):
        try:
            if raw.endswith("Z"):
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
    tokens = list(tokenize(s))
    p = Parser(tokens)
    val = p.parse()
    return val

def load(fp) -> Any:
    text = fp.read()
    return loads(text)

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
