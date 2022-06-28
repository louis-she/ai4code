import re
import tokenize
import io
from nltk.stem import WordNetLemmatizer


stemmer = WordNetLemmatizer()


def links_to_word(text):
    return re.sub("https?:\/\/[^\s]+", " link ", text)


def no_char(text):
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+[a-zA-Z]$", " ", text)
    return text


def no_markdown_special(text):
    return re.sub(r"[\.\*\+\-\_\>\<\~\(\)\[\]]", " ", text)


def no_html_tags(text):
    return re.sub("<.*?>", " ", text)


def no_multi_spaces(text):
    return re.sub(r"\s+", " ", text, flags=re.I)


def lemmatize(text):
    tokens = text.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def underscore_to_space(text: str):
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


def code_preprocess_v4(code):
    code = links_to_word(code)
    code = underscore_to_space(code)
    code = no_char(code)
    code = no_multi_spaces(code)
    code = lemmatize(code)
    return code


def code_preprocess_v7(code):
    # code 最小修改
    code = links_to_word(code)
    code = lemmatize(code)
    return code


def markdown_preprocess_v4(code):
    code = links_to_word(code)
    code = no_markdown_special(code)
    code = no_html_tags(code)
    code = no_char(code)
    code = no_multi_spaces(code)
    code = lemmatize(code)
    return code


def no_markdown_special_v2(text):
    # 保留顶格的 + - * >, 删除其他的
    text = text[0] + re.sub(r"(?<!\n)[\*\+\-\>]", " ", text[1:])

    # 删除 ( ) [ ] ` ~ |
    text = re.sub(r"\(\)\[\]\{\}\<\>\~\|\`\.", " ", text)
    return text


def markdown_preprocess_v6(code):
    """compare to v4:
    1. 不删除单个字符
    2. 保留部分顶格的特殊字符
    3. 先删除 html tags ，再删 markdown 记号
    """
    code = links_to_word(code)
    code = no_html_tags(code)
    code = no_markdown_special_v2(code)
    code = no_multi_spaces(code)
    code = lemmatize(code)
    return code


def markdown_preprocess_v7(code):
    """换行符替换为 [unused1]
    """
    code = code.replace("\n", "[unused1]")
    code = markdown_preprocess_v6(code)
    return code


def code_preprocess_v5(code):
    """
    仅保留顶层代码，丢弃所有 nested 的代码
    """
    lines = code.split("\n")
    outputs = []
    for i, line in enumerate(lines):
        if not line.startswith(" "):
            outputs.append(line)

    return code_preprocess_v4("\n".join(outputs))


def code_preprocess_v6(code):
    """Tokenizer"""
    try:
        code_text = tokenize.generate_tokens(io.StringIO(code).readline)
        return " ".join(
            [tok.string for tok in code_text if tok.type == 1 or tok.type == 55]
        )
    except:
        # 有可能会失败，失败的话 fallback 到 code_preprocess_v4
        return code_preprocess_v4(code)


def preprocessor_v4(text, type):
    """follow mine mind version : )"""
    return dict(code=code_preprocess_v4, markdown=markdown_preprocess_v4)[type](text)


def preprocessor_v5(text, type):
    """代码仅保留最外层
    掉分！
    """
    return dict(code=code_preprocess_v5, markdown=markdown_preprocess_v4)[type](text)


def preprocessor_v6(text, type):
    return dict(code=code_preprocess_v4, markdown=markdown_preprocess_v6)[type](text)


def preprocessor_v7(text, type):
    return dict(code=code_preprocess_v6, markdown=markdown_preprocess_v6)[type](text)


def preprocessor_v8(text, type):
    return dict(code=code_preprocess_v7, markdown=markdown_preprocess_v6)[type](text)


def preprocessor_v9(text, type):
    return dict(code=code_preprocess_v7, markdown=markdown_preprocess_v7)[type](text)
