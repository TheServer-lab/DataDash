# deta_dash.py (fixed: support C-style block comments so parentheses params aren't broken by '/* ... */')
import re
import io
import sys
import copy
import ast
import base64
import uuid
from datetime import datetime, timezone
from typing import Any, Tuple
import operator

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
# DATE must come before NUMBER so ISO datetimes are captured
# Add COMMENT3 to support C-style block comments (/* ... */) so they don't break parsing
TOKEN_REGEX = re.compile(
    r'''
    (?P<WHITESPACE>\s+)
  | (?P<COMMENT>//[^\n]*)
  | (?P<COMMENT2>\#[^\n]*)
  | (?P<COMMENT3>/\*.*?\*/)
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
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<DOT>\.)
  | (?P<PLUS>\+)
  | (?P<MINUS>-)
  | (?P<SLASH>/)
  | (?P<PERCENT>%)
  | (?P<CARET>\^)
  | (?P<DATE>\d{4}-\d{2}-\d{2}T[0-9:\.\+\-Zz]+)
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
            end_idx = text.find(quote, pos + 3)
            if end_idx == -1:
                raise DetaDashParseError("Unterminated triple-quoted string", line, col)
            content = text[pos+3:end_idx]
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
        if isinstance(n, (ast.Call, ast.Import, ast.ImportFrom, ast.Lambda, ast.DictComp, ast.ListComp, ast.GeneratorExp, ast.Yield, ast.YieldFrom)):
            raise DetaDashError("Disallowed expression construct")

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
    }
    un_ops = {
        ast.UAdd: lambda x: x,
        ast.USub: operator.neg,
    }

    NumType = getattr(ast, "Num", None)
    IndexType = getattr(ast, "Index", None)

    def eval_node(n):
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if NumType is not None and isinstance(n, NumType):
            return n.n
        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)
            op_type = type(n.op)
            if op_type in ops:
                try:
                    return ops[op_type](left, right)
                except Exception as e:
                    raise DetaDashError(f"Error in binary operation: {e}")
            raise DetaDashError(f"Unsupported binary operator: {op_type}")
        if isinstance(n, ast.UnaryOp):
            operand = eval_node(n.operand)
            op_type = type(n.op)
            if op_type in un_ops:
                try:
                    return un_ops[op_type](operand)
                except Exception as e:
                    raise DetaDashError(f"Error in unary operation: {e}")
            raise DetaDashError(f"Unsupported unary operator: {op_type}")
        if isinstance(n, ast.Name):
            if n.id not in context:
                raise DetaDashError(f"Unknown name in expression: {n.id!r}")
            return context[n.id]
        if isinstance(n, ast.Attribute):
            val = eval_node(n.value)
            attr = n.attr
            if isinstance(val, dict):
                if attr not in val:
                    raise DetaDashError(f"Unknown attribute {attr!r} on object")
                return val[attr]
            try:
                return getattr(val, attr)
            except Exception:
                raise DetaDashError(f"Cannot access attribute {attr!r}")
        if isinstance(n, ast.Subscript):
            val = eval_node(n.value)
            sl = n.slice
            if IndexType is not None and isinstance(sl, IndexType):
                idx = eval_node(sl.value)
            else:
                try:
                    idx = eval_node(sl)
                except DetaDashError:
                    raise DetaDashError("Unsupported subscript slice type")
            try:
                return val[idx]
            except Exception as e:
                raise DetaDashError(f"Subscript error: {e}")
        if isinstance(n, ast.Tuple):
            return tuple(eval_node(elt) for elt in n.elts)
        if isinstance(n, ast.List):
            return [eval_node(elt) for elt in n.elts]
        raise DetaDashError(f"Unsupported expression node: {type(n).__name__}")

    try:
        result = eval_node(node)
        if isinstance(result, (int, float)):
            return result
        if isinstance(result, bool):
            return result
        try:
            return float(result)
        except Exception:
            raise DetaDashError(f"Expression did not evaluate to a number: {expr_src!r}")
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
            if key_token.type not in ("IDENT", "STRING", "TRI_STRING"):
                raise DetaDashParseError("Expected key (identifier or string) in object", key_token.line, key_token.col)
            key = key_token.value
            self.expect("COLON")
            value = self.parse_value(allow_anchor=True)
            obj[key] = value
            self.root_context[key] = value
            if self.accept("COMMA"):
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
            # collect expression tokens, but keep track of bracket/paren depth so
            # subscripts and grouped expressions are fully included
            self.next()  # consume '='
            expr_parts = []
            depth = 0
            while True:
                nt = self.peek()
                # break only on comma/closing-brace/EOF when not inside parens/brackets
                if nt.type in ("COMMA", "RBRACE", "EOF") and depth == 0:
                    break
                tok = self.next()
                expr_parts.append(tok.value)
                if tok.type in ("LPAREN", "LBRACK"):
                    depth += 1
                elif tok.type in ("RPAREN", "RBRACK"):
                    if depth > 0:
                        depth -= 1
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
                if self.peek().type == "LPAREN":
                    # collect everything up to the matching RPAREN (allowing comments to be skipped)
                    self.next()  # consume LPAREN
                    param_parts = []
                    depth = 1
                    while True:
                        nt = self.peek()
                        if nt.type == "EOF":
                            raise DetaDashParseError("Unterminated parentheses in @ tag", nt.line, nt.col)
                        tok = self.next()
                        if tok.type == "LPAREN":
                            depth += 1
                        elif tok.type == "RPAREN":
                            depth -= 1
                            if depth == 0:
                                break
                        param_parts.append(tok.value)
                    param = "".join(str(x) for x in param_parts).strip()
                    # strip surrounding quotes if the whole param was a quoted string
                    if len(param) >= 2 and ((param[0] == '"' and param[-1] == '"') or (param[0] == "'" and param[-1] == "'")):
                        param = param[1:-1]
                    return self.handle_at_tag(ident, param)
                elif self.peek().type == "STRING":
                    param = self.next().value
                    return self.handle_at_tag(ident, param)
                else:
                    raise DetaDashParseError("Unsupported @ literal form", next_token.line, next_token.col)
            elif next_token.type in ("STRING", "TRI_STRING"):
                return next_token.value
            elif next_token.type in ("DATE", "NUMBER"):
                raw = next_token.value
                return self.handle_at_date_literal(raw)
            else:
                raise DetaDashParseError("Unsupported @ literal token", next_token.line, next_token.col)
        if t.type == "BANG":
            self.next()
            next_token = self.peek()
            if next_token.type == "IDENT":
                ident = self.next().value
                if self.peek().type == "LPAREN":
                    # collect everything up to the matching RPAREN for !tag(...)
                    self.next()  # consume LPAREN
                    param_parts = []
                    depth = 1
                    while True:
                        nt = self.peek()
                        if nt.type == "EOF":
                            raise DetaDashParseError("Unterminated parentheses in ! tag", nt.line, nt.col)
                        tok = self.next()
                        if tok.type == "LPAREN":
                            depth += 1
                        elif tok.type == "RPAREN":
                            depth -= 1
                            if depth == 0:
                                break
                        param_parts.append(tok.value)
                    param = "".join(str(x) for x in param_parts).strip()
                    if len(param) >= 2 and ((param[0] == '"' and param[-1] == '"') or (param[0] == "'" and param[-1] == "'")):
                        param = param[1:-1]
                    return self.handle_bang_tag(ident, param)
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
        if t.type == "DATE":
            self.next()
            return self.handle_at_date_literal(t.value)
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
            if isinstance(raw, str) and raw.endswith("Z"):
                iso = raw[:-1] + "+00:00"
                return datetime.fromisoformat(iso)
            return datetime.fromisoformat(raw)
        except Exception:
            raise DetaDashParseError(f"Invalid date literal: {raw}")

    def handle_bang_tag(self, ident, param):
        if ident.lower() == "base64":
            try:
                if isinstance(param, str):
                    return base64.b64decode(param)
                return base64.b64decode(str(param))
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
  defaults: &defaults { window: { width: 1280, height: 720 } },
  store_cfg: &store {
    inventory: [
      { id: "sword", price: 150, qty: 3 },
      { id: "shield", price: 100, qty: 2 },
      { id: "potion", price: 25, qty: 10 },
    ],
  },
  total_inventory_value: = store_cfg.inventory[0].price * store_cfg.inventory[0].qty
                         + store_cfg.inventory[1].price * store_cfg.inventory[1].qty
                         + store_cfg.inventory[2].price * store_cfg.inventory[2].qty,
  adjusted_area: = (defaults.window.width - 200) * (defaults.window.height - 100),
  logo_png: !base64("iVBORw0KGgoAAAANSUhEUgAAAAUA" /* truncated example data */),
}
'''
    print("Parsing example with subscripts, parentheses and block comments in tags...")
    try:
        data = loads(example)
        from pprint import pprint
        pprint(data)
        print("\nSerialized back to DetaDash-like text:")
        print(dumps(data))
    except Exception as e:
        print("Error:", e)
