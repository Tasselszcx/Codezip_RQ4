import os
import math
import sys
import argparse
import time
from typing import List, Tuple, Dict, Optional, Callable
from PIL import Image as PIL_Image, ImageDraw, ImageFont
import tiktoken

# 尝试导入 pygments（用于代码高亮）
try:
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
    from pygments.token import Token

    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    print("警告: pygments 未安装，将无法使用代码高亮功能")

# 尝试导入 transformers（用于 AutoProcessor）
try:
    from transformers import AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 未安装，将无法使用 AutoProcessor 计算token")


# 换行符标记（用于在图片中保留换行符信息但不实际换行）
NEWLINE_MARKER = "⏎"  # 使用可见的换行符标记

# 压缩比列表（用于 resize 模式）
COMPRESSION_RATIOS = [0.5, 1, 1.5, 2, 4, 8]


def get_all_modes():
    """
    获取所有模式列表，包括 text_only, image 和所有压缩比模式
    """
    modes = ["text_only", "image"]
    for ratio in sorted(COMPRESSION_RATIOS):
        modes.append(f"image_ratio{ratio}")
    return modes


def get_flat_filename(filename: str) -> str:
    """将原始文件名转换为扁平化格式（用于文件命名）"""
    if filename is None:
        return "unknown"
    return filename.replace("/", "_")


def get_font(font_size: int, font_path: str = None):
    """
    获取Monospace字体对象（优先支持中文）

    Args:
        font_size: 字体大小（像素）
        font_path: 字体路径（可选，如果为None则自动查找系统Monospace字体）

    Returns:
        ImageFont对象（Monospace字体）
    """
    try:
        if font_path and os.path.exists(font_path):
            # 使用指定字体
            font = ImageFont.truetype(font_path, font_size)
            print(f"  使用指定字体: {font_path}")
            return font
        else:
            # 🌟 优先使用支持中文的等宽字体（Windows系统）
            monospace_font_paths = [
                # Windows 中文等宽字体（优先）
                "C:/Windows/Fonts/simhei.ttf",      # 黑体（支持中文）
                "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑（支持中文，常用）
                "C:/Windows/Fonts/simsun.ttc",      # 宋体（支持中文）
                "C:/Windows/Fonts/STXIHEI.TTF",     # 华文细黑（macOS/Windows）
                "C:/Windows/Fonts/simkai.ttf",      # 楷体（支持中文）
                # Windows 英文等宽字体（备选）
                "C:/Windows/Fonts/consola.ttf",     # Consolas（英文等宽）
                "C:/Windows/Fonts/cour.ttf",        # Courier New
                "C:/Windows/Fonts/courbd.ttf",
                "C:/Windows/Fonts/lucon.ttf",       # Lucida Console
                # Linux 支持中文的字体
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",       # 文泉驿微米黑
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",         # 文泉驿正黑
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                # Linux 英文等宽字体（备选）
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
                # macOS 中文字体
                "/System/Library/Fonts/STHeiti Light.ttc",      # 华文黑体
                "/System/Library/Fonts/PingFang.ttc",           # 苹方（支持中文）
                "/Library/Fonts/Arial Unicode.ttf",             # Arial Unicode MS（支持多语言）
                # macOS 英文等宽字体（备选）
                "/System/Library/Fonts/Menlo.ttc",
                "/Library/Fonts/Courier New.ttf",
            ]

            font = None
            used_font = None
            for path in monospace_font_paths:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, font_size)
                        used_font = path
                        break
                    except Exception as e:
                        # 如果加载失败，继续尝试下一个
                        continue

            if font is None:
                # 如果所有字体都找不到，尝试使用PIL默认字体
                font = ImageFont.load_default()
                print("  警告: 未找到合适的TrueType字体，使用PIL默认字体（可能不支持中文）")
            else:
                print(f"  使用字体: {os.path.basename(used_font)}")

            return font
    except Exception as e:
        print(f"  警告: 加载字体失败: {e}，使用默认字体")
        return ImageFont.load_default()


def prepare_text_for_rendering(text: str, preserve_newlines: bool = False) -> str:
    """
    准备文本用于渲染：
    1. 如果 preserve_newlines=False：替换换行符为可见标记（⏎），保留换行符信息但不实际换行
    2. 如果 preserve_newlines=True：保留换行符，用于正常换行模式
    3. 替换制表符为空格（避免显示为方块字符）
    4. 处理特殊字符

    Args:
        text: 要处理的文本
        preserve_newlines: 是否保留换行符（True=正常换行模式，False=紧凑模式）
    """
    # 替换制表符为4个空格（代码中通常使用4个空格作为缩进）
    text = text.replace("\t", "    ")  # 4个空格

    if not preserve_newlines:
        # 紧凑模式：替换换行符为可见标记（保留换行符信息但不实际换行）
        text = text.replace("\n", NEWLINE_MARKER)

    # 处理特殊字符（统一为ASCII字符，避免字体问题）
    typographic_replacements = {
        "'": "'",
        "'": "'",
        '"': '"',
        '"': '"',
        "–": "-",
        "—": "--",
        "…": "...",
    }
    for original, replacement in typographic_replacements.items():
        text = text.replace(original, replacement)

    return text


def crop_whitespace(
    img: PIL_Image.Image, bg_color: str = "white", keep_margin: Tuple[int, int] = (0, 0)
) -> PIL_Image.Image:
    """
    裁剪图片的白边，去掉右边和下面的空白区域

    Args:
        img: PIL Image对象
        bg_color: 背景颜色（用于检测白边）
        keep_margin: 保留的边距 (left, top)，默认(0, 0)

    Returns:
        裁剪后的PIL Image对象
    """
    # 转换为灰度图以便检测
    gray = img.convert("L")

    # 将背景颜色转换为灰度值
    if bg_color == "white":
        bg_threshold = 240  # 白色阈值
    elif bg_color == "black":
        bg_threshold = 15  # 黑色阈值
    else:
        # 对于其他颜色，使用RGB值计算灰度
        try:
            from PIL import ImageColor

            rgb = ImageColor.getrgb(bg_color)
            bg_threshold = int(sum(rgb) / 3)  # 简单平均
        except:
            bg_threshold = 240  # 默认白色

    # 创建掩码：非背景区域为1，背景区域为0
    mask = gray.point(lambda p: 0 if p > bg_threshold else 255, mode="1")

    # 获取内容边界框
    bbox = mask.getbbox()

    if bbox is None:
        # 如果没有内容，返回原图
        return img

    # bbox格式: (left, top, right, bottom)
    left, top, right, bottom = bbox

    # 保留左边和上边的边距
    left = max(0, left - keep_margin[0])
    top = max(0, top - keep_margin[1])

    # 裁剪图片（保留左边和上边的边距，裁剪右边和下面的空白）
    cropped = img.crop((left, top, right, bottom))

    return cropped


def parse_code_with_syntax_highlighting(
    code: str, filename: str = None, language: str = None, theme: str = "light"
) -> List[Tuple[str, str]]:
    """
    使用 Pygments 解析代码，返回带颜色的token列表

    Args:
        code: 源代码文本
        filename: 文件名（用于自动检测语言）
        language: 语言名称（如 'python', 'javascript' 等），如果为None则自动检测
        theme: 主题名称 ('light' 或 'modern')

    Returns:
        List of (text, color) tuples，每个tuple包含文本内容和对应的颜色（RGB格式）
    """
    if not PYGMENTS_AVAILABLE:
        # 如果没有安装 Pygments，返回单色文本
        return [(code, "#000000")]

    try:
        # 确定语言
        if language:
            lexer = get_lexer_by_name(language)
        elif filename:
            try:
                lexer = guess_lexer_for_filename(filename, code)
                try:
                    from pygments.lexers.special import TextLexer
                except Exception:
                    TextLexer = None

                # 如果按文件名判断得到的是纯文本lexer（如 .txt），尝试基于内容再猜一次
                if TextLexer is not None and isinstance(lexer, TextLexer):
                    try:
                        from pygments.lexers import guess_lexer

                        content_lexer = guess_lexer(code)
                        if not isinstance(content_lexer, TextLexer):
                            lexer = content_lexer
                    except Exception:
                        pass
            except:
                try:
                    from pygments.lexers import guess_lexer

                    lexer = guess_lexer(code)
                except:
                    lexer = get_lexer_by_name("python")
        else:
            try:
                from pygments.lexers import guess_lexer

                lexer = guess_lexer(code)
            except:
                lexer = get_lexer_by_name("python")

        # 定义颜色映射
        if theme == "modern" or theme == "morden":
            # VS Code Light Modern 主题
            color_map = {
                # Control Flow Keywords (Purple)
                Token.Keyword: "#AF00DB",             # Default Keyword -> Purple (covers if, return, import, from, etc.)
                Token.Keyword.Namespace: "#AF00DB",   # import/from -> Purple
                
                # Declaration/Storage Keywords (Blue)
                Token.Keyword.Declaration: "#0000FF", # def, class -> Blue
                Token.Keyword.Type: "#0000FF",        # int, str -> Blue
                Token.Keyword.Constant: "#0000FF",    # True, False, None -> Blue
                Token.Operator.Word: "#0000FF",       # and, or, not, is, in -> Blue
                
                # Functions & Builtins (Yellow/Ochre)
                Token.Name.Function: "#795E26",       # Function definitions -> Ochre
                Token.Name.Builtin: "#795E26",        # Built-in functions (open, print) -> Ochre
                Token.Name.Builtin.Pseudo: "#0000FF", # self, cls -> Blue (VS Code treats self as variable or keyword depending on config, but standard is often Blue)
                
                # Classes (Teal)
                Token.Name.Class: "#267F99",          # Class names -> Teal
                
                # Variables (Dark Blue / Light Blue)
                Token.Name: "#000000",                # Default Name -> Black
                Token.Name.Variable: "#001080",       # Variables -> Dark Blue
                Token.Name.Variable.Instance: "#001080",
                Token.Name.Variable.Class: "#001080",
                Token.Name.Variable.Global: "#001080",
                Token.Name.Attribute: "#001080",      # Attributes -> Dark Blue
                
                # Strings (Red)
                Token.String: "#A31515",              # Strings -> Red
                Token.String.Doc: "#008000",          # Docstrings -> Green (VS Code convention)
                Token.String.Interpol: "#A31515",
                Token.String.Escape: "#EE0000",
                
                # Numbers (Green)
                Token.Number: "#098658",              # Numbers -> Green
                Token.Number.Integer: "#098658",
                Token.Number.Float: "#098658",
                Token.Number.Hex: "#098658",
                
                # Comments (Green)
                Token.Comment: "#008000",             # Comments -> Green
                Token.Comment.Single: "#008000",
                Token.Comment.Multiline: "#008000",
                
                # Operators & Punctuation
                Token.Operator: "#000000",            # Operators -> Black
                Token.Punctuation: "#000000",         # Punctuation -> Black
                
                # Others
                Token.Error: "#FF0000",
                Token.Generic.Deleted: "#A31515",
                Token.Generic.Inserted: "#008000",
            }
        else:
            # 默认主题 (类似 VS Code Classic Light)
            color_map = {
                Token.Keyword: "#0000FF",  # 关键字：蓝色
                Token.Keyword.Constant: "#0000FF",  # 常量关键字：蓝色
                Token.Keyword.Declaration: "#0000FF",  # 声明关键字：蓝色
                Token.Keyword.Namespace: "#0000FF",  # 命名空间关键字：蓝色
                Token.Keyword.Pseudo: "#0000FF",  # 伪关键字：蓝色
                Token.Keyword.Reserved: "#0000FF",  # 保留关键字：蓝色
                Token.Keyword.Type: "#0000FF",  # 类型关键字：蓝色
                Token.Name: "#000000",  # 名称：黑色
                Token.Name.Builtin: "#795E26",  # 内置名称：棕色
                Token.Name.Class: "#267F99",  # 类名：青色
                Token.Name.Function: "#795E26",  # 函数名：棕色
                Token.Name.Namespace: "#000000",  # 命名空间：黑色
                Token.String: "#A31515",  # 字符串：红色
                Token.String.Doc: "#008000",  # 文档字符串：绿色
                Token.String.Escape: "#A31515",  # 转义字符：红色
                Token.String.Interpol: "#A31515",  # 插值字符串：红色
                Token.String.Other: "#A31515",  # 其他字符串：红色
                Token.String.Regex: "#811F3F",  # 正则表达式：深红色
                Token.String.Symbol: "#A31515",  # 符号字符串：红色
                Token.Number: "#098658",  # 数字：绿色
                Token.Number.Bin: "#098658",  # 二进制数：绿色
                Token.Number.Float: "#098658",  # 浮点数：绿色
                Token.Number.Hex: "#098658",  # 十六进制数：绿色
                Token.Number.Integer: "#098658",  # 整数：绿色
                Token.Number.Long: "#098658",  # 长整数：绿色
                Token.Number.Oct: "#098658",  # 八进制数：绿色
                Token.Comment: "#008000",  # 注释：绿色
                Token.Comment.Hashbang: "#008000",  # Hashbang注释：绿色
                Token.Comment.Multiline: "#008000",  # 多行注释：绿色
                Token.Comment.Single: "#008000",  # 单行注释：绿色
                Token.Comment.Special: "#008000",  # 特殊注释：绿色
                Token.Operator: "#000000",  # 运算符：黑色
                Token.Operator.Word: "#0000FF",  # 运算符关键字：蓝色
                Token.Punctuation: "#000000",  # 标点符号：黑色
                Token.Error: "#FF0000",  # 错误：红色
                Token.Generic: "#000000",  # 通用：黑色
                Token.Generic.Deleted: "#A31515",  # 删除：红色
                Token.Generic.Emph: "#000000",  # 强调：黑色
                Token.Generic.Error: "#FF0000",  # 错误：红色
                Token.Generic.Heading: "#000000",  # 标题：黑色
                Token.Generic.Inserted: "#008000",  # 插入：绿色
                Token.Generic.Output: "#000000",  # 输出：黑色
                Token.Generic.Prompt: "#000000",  # 提示：黑色
                Token.Generic.Strong: "#000000",  # 粗体：黑色
                Token.Generic.Subheading: "#000000",  # 子标题：黑色
                Token.Generic.Traceback: "#000000",  # 跟踪：黑色
                Token.Other: "#000000",  # 其他：黑色
                Token.Text: "#000000",  # 文本：黑色
                Token.Text.Whitespace: "#000000",  # 空白：黑色
            }

        # 解析代码
        tokens = list(lex(code, lexer))
        result = []

        for token_type, text in tokens:
            # 获取颜色
            color = "#000000"  # 默认黑色
            for token_class, mapped_color in color_map.items():
                if token_type in token_class:
                    color = mapped_color
                    break

            result.append((text, color))

        return result

    except Exception as e:
        # 如果解析失败，返回单色文本
        print(f"警告: 代码高亮解析失败: {e}，使用单色模式")
        return [(code, "#000000")]


def text_to_image_compact(
    text: str,
    width: int = 800,
    height: int = 1200,
    font_size: int = 10,
    line_height: float = 1.2,
    margin_px: int = 10,
    dpi: int = 300,
    font_path: str = None,
    bg_color: str = "white",
    text_color: str = "black",
    preserve_newlines: bool = False,
    enable_syntax_highlight: bool = False,
    filename: str = None,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    enable_bold: bool = False,
    theme: str = "light",
) -> List[PIL_Image.Image]:
    """
    将文本渲染为紧凑型图片（使用PIL直接渲染，快速且精确控制）
    支持自动分页，当内容超过一张图片时自动生成多张

    Args:
        text: 要渲染的文本
        width: 图片宽度（像素）
        height: 图片高度（像素）
        font_size: 字体大小（像素）
        line_height: 行高（倍数，如1.2表示1.2倍字体大小）
        margin_px: 边距（像素）
        dpi: DPI设置（用于元数据，不影响实际像素尺寸）
        font_path: 字体路径（可选）
        bg_color: 背景颜色
        text_color: 文字颜色
        preserve_newlines: 是否保留换行符（True=正常换行模式，False=紧凑模式）
        enable_syntax_highlight: 是否启用语法高亮（需要安装 pygments）
        filename: 文件名（用于自动检测语言，仅在启用语法高亮时使用）
        language: 语言名称（如 'python', 'javascript' 等），仅在启用语法高亮时使用
        crop_whitespace: 是否裁剪白边（True=裁剪，False=保留原始尺寸）
        enable_two_column: 是否启用两列布局（True=当左列写满高度且宽度<=图片一半时，切换到右列继续写入，False=单列布局）
        enable_bold: 是否加粗（True=所有文本加粗，False=正常）
        theme: 语法高亮主题 ('light' 或 'modern')

    Returns:
        PIL Image对象列表（可能有多张图片）
    """
    # 获取字体（用于测量）
    temp_img = PIL_Image.new("RGB", (width, height), color=bg_color)
    temp_draw = ImageDraw.Draw(temp_img)
    font = get_font(font_size, font_path)

    # 计算实际可用区域
    text_area_width = width - 2 * margin_px
    text_area_height = height - 2 * margin_px

    # 计算行高（像素）
    line_height_px = int(font_size * line_height)

    # 计算每张图片能容纳的行数
    max_lines_per_page = (
        int(text_area_height / line_height_px) if line_height_px > 0 else 1
    )

    # 如果启用语法高亮，解析代码获取带颜色的token列表
    if enable_syntax_highlight and PYGMENTS_AVAILABLE:
        # 使用原始文本进行语法高亮解析（在prepare_text_for_rendering之前）
        colored_tokens = parse_code_with_syntax_highlighting(
            text, filename=filename, language=language, theme=theme
        )

        # 准备文本（处理制表符和特殊字符）
        processed_tokens = []
        for token_text, token_color in colored_tokens:
            # 处理制表符
            processed_token_text = token_text.replace("\t", "    ")
            # 处理特殊字符
            typographic_replacements = {
                "'": "'",
                "'": "'",
                '"': '"',
                '"': '"',
                "–": "-",
                "—": "--",
                "…": "...",
            }
            for original, replacement in typographic_replacements.items():
                processed_token_text = processed_token_text.replace(
                    original, replacement
                )

            # 处理换行符
            if preserve_newlines:
                # 保留换行符
                processed_tokens.append((processed_token_text, token_color))
            else:
                # 替换换行符为可见标记
                processed_tokens.append(
                    (processed_token_text.replace("\n", NEWLINE_MARKER), token_color)
                )

        # 使用彩色token进行渲染
        pages = []
        current_page_lines = 0
        current_x = margin_px
        current_y = margin_px
        current_column = 0  # 0=左列, 1=右列
        max_column_width = 0  # 当前列的最大宽度
        column_start_x = margin_px  # 当前列的起始x位置
        # 根据 enable_two_column 决定列宽度
        if enable_two_column:
            column_width = (width - 2 * margin_px) // 2  # 每列的可用宽度（减去列间距）
            column_gap = 10  # 两列之间的间距
        else:
            column_width = width - 2 * margin_px  # 单列模式，使用整个宽度
            column_gap = 0

        # 创建第一页
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        for token_text, token_color in processed_tokens:
            try:
                from PIL import ImageColor

                rgb_color = ImageColor.getrgb(token_color)
            except:
                rgb_color = ImageColor.getrgb("#000000")  # 默认黑色

            for char in token_text:
                if preserve_newlines and char == "\n":
                    max_column_width = max(max_column_width, current_x - column_start_x)
                    current_y += line_height_px
                    current_x = column_start_x
                    current_page_lines += 1

                    if current_page_lines >= max_lines_per_page:
                        if (
                            enable_two_column
                            and current_column == 0
                            and max_column_width <= (width / 2)
                        ):
                            current_column = 1
                            column_start_x = width // 2 + column_gap // 2
                            current_x = column_start_x
                            current_y = margin_px
                            current_page_lines = 0
                            max_column_width = 0
                        else:
                            if should_crop_whitespace:
                                img = crop_whitespace(
                                    img, bg_color, keep_margin=(margin_px, margin_px)
                                )
                            if dpi:
                                img.info["dpi"] = (dpi, dpi)
                            pages.append(img)

                            img = PIL_Image.new("RGB", (width, height), color=bg_color)
                            draw = ImageDraw.Draw(img)
                            current_page_lines = 0
                            current_y = margin_px
                            current_column = 0
                            column_start_x = margin_px
                            current_x = margin_px
                            max_column_width = 0
                    continue

                try:
                    char_w = temp_draw.textlength(char, font=font)
                except:
                    char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                    char_w = char_bbox[2] - char_bbox[0]

                if enable_two_column:
                    column_right_bound = (
                        column_start_x + column_width
                        if current_column == 0
                        else width - margin_px
                    )
                else:
                    column_right_bound = width - margin_px
                if (
                    current_x + char_w > column_right_bound
                    and current_x > column_start_x
                ):
                    max_column_width = max(max_column_width, current_x - column_start_x)
                    current_y += line_height_px
                    current_x = column_start_x
                    current_page_lines += 1

                    if current_page_lines >= max_lines_per_page:
                        if (
                            enable_two_column
                            and current_column == 0
                            and max_column_width <= (width / 2)
                        ):
                            current_column = 1
                            column_start_x = width // 2 + column_gap // 2
                            current_x = column_start_x
                            current_y = margin_px
                            current_page_lines = 0
                            max_column_width = 0
                        else:
                            if should_crop_whitespace:
                                img = crop_whitespace(
                                    img, bg_color, keep_margin=(margin_px, margin_px)
                                )
                            if dpi:
                                img.info["dpi"] = (dpi, dpi)
                            pages.append(img)

                            img = PIL_Image.new("RGB", (width, height), color=bg_color)
                            draw = ImageDraw.Draw(img)
                            current_page_lines = 0
                            current_y = margin_px
                            current_column = 0
                            column_start_x = margin_px
                            current_x = margin_px
                            max_column_width = 0

                if enable_bold:
                    for dx, dy in [(0, 0), (1, 0)]:
                        draw.text((current_x + dx, current_y + dy), char, font=font, fill=rgb_color)
                else:
                    draw.text((current_x, current_y), char, font=font, fill=rgb_color)
                current_x += char_w
                max_column_width = max(max_column_width, current_x - column_start_x)

        if current_page_lines > 0 or current_x > margin_px:
            if should_crop_whitespace:
                img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
            if dpi:
                img.info["dpi"] = (dpi, dpi)
            pages.append(img)

        # 计算并打印生成时间
        result_pages = pages if pages else [img]
        return result_pages

    else:
        processed_text = prepare_text_for_rendering(
            text, preserve_newlines=preserve_newlines
        )

        lines = []

        if preserve_newlines:
            original_lines = processed_text.split("\n")
            for original_line in original_lines:
                current_line = ""
                current_line_width = 0

                for char in original_line:
                    try:
                        char_w = temp_draw.textlength(char, font=font)
                    except:
                        char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                        char_w = char_bbox[2] - char_bbox[0]

                    if current_line_width + char_w > text_area_width and current_line:
                        lines.append(current_line)
                        current_line = char
                        current_line_width = char_w
                    else:
                        current_line += char
                        current_line_width += char_w

                if current_line:
                    lines.append(current_line)
        else:
            current_line = ""
            current_line_width = 0

            for char in processed_text:
                try:
                    char_w = temp_draw.textlength(char, font=font)
                except:
                    char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                    char_w = char_bbox[2] - char_bbox[0]

                if current_line_width + char_w > text_area_width and current_line:
                    lines.append(current_line)
                    current_line = char
                    current_line_width = char_w
                else:
                    current_line += char
                    current_line_width += char_w

            if current_line:
                lines.append(current_line)

        total_lines = len(lines)
        pages = []

        if enable_two_column:
            column_width = (width - 2 * margin_px) // 2
            column_gap = 10
            left_column_start = margin_px
            right_column_start = width // 2 + column_gap // 2
            left_column_right_bound = left_column_start + column_width
            right_column_right_bound = width - margin_px
        else:
            column_width = width - 2 * margin_px
            column_gap = 0
            left_column_start = margin_px
            right_column_start = margin_px
            left_column_right_bound = width - margin_px
            right_column_right_bound = width - margin_px

        page_start = 0
        while page_start < total_lines:
            img = PIL_Image.new("RGB", (width, height), color=bg_color)
            draw = ImageDraw.Draw(img)

            x = left_column_start
            y = margin_px
            current_page_lines = 0
            max_left_column_width = 0

            while page_start < total_lines and current_page_lines < max_lines_per_page:
                line = lines[page_start]

                current_line_x = x
                line_chars = list(line)
                line_drawn = False

                for char in line_chars:
                    try:
                        char_w = temp_draw.textlength(char, font=font)
                    except:
                        char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                        char_w = char_bbox[2] - char_bbox[0]

                    if (
                        current_line_x + char_w > left_column_right_bound
                        and current_line_x > left_column_start
                    ):
                        y += line_height_px
                        current_line_x = left_column_start
                        current_page_lines += 1
                        if current_page_lines >= max_lines_per_page:
                            break

                    if enable_bold:
                        for dx, dy in [(0, 0), (1, 0)]:
                            draw.text((current_line_x + dx, y + dy), char, font=font, fill=text_color)
                    else:
                        draw.text((current_line_x, y), char, font=font, fill=text_color)
                    current_line_x += char_w
                    max_left_column_width = max(
                        max_left_column_width, current_line_x - left_column_start
                    )
                    line_drawn = True

                if line_drawn:
                    y += line_height_px
                    x = left_column_start
                    current_page_lines += 1
                    page_start += 1
                else:
                    page_start += 1
                    break

            if (
                enable_two_column
                and current_page_lines >= max_lines_per_page
                and max_left_column_width <= (width / 2)
            ):
                x = right_column_start
                y = margin_px
                current_page_lines = 0

                while (
                    page_start < total_lines and current_page_lines < max_lines_per_page
                ):
                    line = lines[page_start]

                    current_line_x = x
                    line_chars = list(line)
                    line_drawn = False

                    for char in line_chars:
                        try:
                            char_w = temp_draw.textlength(char, font=font)
                        except:
                            char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                            char_w = char_bbox[2] - char_bbox[0]

                        if (
                            current_line_x + char_w > right_column_right_bound
                            and current_line_x > right_column_start
                        ):
                            y += line_height_px
                            current_line_x = right_column_start
                            current_page_lines += 1
                            if current_page_lines >= max_lines_per_page:
                                break

                        if enable_bold:
                            for dx, dy in [(0, 0), (1, 0)]:
                                draw.text((current_line_x + dx, y + dy), char, font=font, fill=text_color)
                        else:
                            draw.text((current_line_x, y), char, font=font, fill=text_color)
                        current_line_x += char_w
                        line_drawn = True

                    if line_drawn:
                        y += line_height_px
                        x = right_column_start
                        current_page_lines += 1
                        page_start += 1
                    else:
                        page_start += 1
                        break

            if should_crop_whitespace:
                img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))

            if dpi:
                img.info["dpi"] = (dpi, dpi)

            pages.append(img)

        return pages


def generate_images_for_file(
    filename: str,
    source_code: str,
    base_output_dir: str,
    width: int = 800,
    height: int = 1200,
    font_size: int = 10,
    line_height: float = 1.2,
    dpi: int = 300,
    font_path: str = None,
    unique_id: str = None,
    preserve_newlines: bool = False,
    enable_syntax_highlight: bool = False,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    enable_bold: bool = False,
) -> List[str]:
    """
    为指定文件生成紧凑型图片

    Args:
        filename: 原始文件名，如 'src/black/__init__.py'
        source_code: 源代码内容
        base_output_dir: 基础输出目录（会在其中创建分辨率文件夹）
        width: 图片宽度
        height: 图片高度
        font_size: 字体大小
        line_height: 行高
        dpi: DPI设置
        unique_id: 唯一标识符（用于文件命名）
        preserve_newlines: 是否保留换行符
        enable_syntax_highlight: 是否启用语法高亮
        language: 语言名称（如 'python', 'javascript' 等），如果为None则从文件名自动检测
        should_crop_whitespace: 是否裁剪白边（True=裁剪，False=保留原始尺寸）
        enable_two_column: 是否启用两列布局（True=当左列写满高度且宽度<=图片一半时，切换到右列继续写入，False=单列布局）

    Returns:
        图片路径列表
    """
    resolution_parts = [f"{width}x{height}"]
    if enable_syntax_highlight:
        resolution_parts.append("hl")
    if preserve_newlines:
        resolution_parts.append("nl")
    resolution_folder_name = "_".join(resolution_parts)
    resolution_dir = os.path.join(base_output_dir, resolution_folder_name)
    os.makedirs(resolution_dir, exist_ok=True)

    if unique_id is None:
        unique_id = filename.replace("/", "_")

    image_paths = []

    try:
        images = text_to_image_compact(
            source_code,
            width=width,
            height=height,
            font_size=font_size,
            line_height=line_height,
            dpi=dpi,
            font_path=font_path,
            preserve_newlines=preserve_newlines,
            enable_syntax_highlight=enable_syntax_highlight,
            filename=filename,
            language=language,
            should_crop_whitespace=should_crop_whitespace,
            enable_two_column=enable_two_column,
            enable_bold=enable_bold
        )

        if images:
            for page_num, image in enumerate(images, 1):
                image_filename = f"page_{page_num:03d}.png"
                image_path = os.path.join(resolution_dir, image_filename)
                image.save(image_path)
                image_paths.append(os.path.abspath(image_path))
                print(
                    f"  生成图片: {image_filename} ({width}x{height}, font={font_size}px, line-height={line_height})"
                )

            print(f"  共生成 {len(images)} 张图片")
        else:
            raise RuntimeError("无法生成图片")

    except Exception as e:
        print(f"  错误: 生成图片失败: {e}")
        raise

    return image_paths


def calculate_image_tokens_qwen3(width: int, height: int) -> int:
    """
    使用 Qwen3 方法计算图片的 token 数量
    公式: (长/16 * 宽/16)/4

    Args:
        width: 图片宽度
        height: 图片高度

    Returns:
        估算的 token 数量
    """
    tokens = (width / 16 * height / 16) / 4
    return int(tokens)


def find_closest_resolution_prefer_larger(
    target_tokens: int, resolution_list: List[int], tolerance_ratio: float = 1.4
) -> int:
    """
    根据目标 token 数找到最接近的分辨率，优先选择更大的分辨率

    如果多个分辨率的token数都在合理范围内（最大token数 <= 最小token数 * tolerance_ratio），
    则选择最大的分辨率。
    """
    candidates = []
    for resolution in resolution_list:
        tokens = calculate_image_tokens_qwen3(resolution, resolution)
        diff = abs(tokens - target_tokens)
        candidates.append((resolution, tokens, diff))

    candidates.sort(key=lambda x: x[2])

    if not candidates:
        return resolution_list[0]

    min_diff = candidates[0][2]
    threshold_diff = min_diff * 1.2  # 允许差异在最小差异的1.2倍以内

    close_candidates = [c for c in candidates if c[2] <= threshold_diff]

    if len(close_candidates) > 1:
        min_tokens = min(c[1] for c in close_candidates)
        max_tokens = max(c[1] for c in close_candidates)

        if max_tokens <= min_tokens * tolerance_ratio:
            return max(c[0] for c in close_candidates)

    return candidates[0][0]


def resize_images_for_compression(
    images: List[PIL_Image.Image],
    text_tokens: int,
    compression_ratios: List[int] = None,
) -> Dict[int, Tuple[List[PIL_Image.Image], int]]:
    """
    根据压缩比将图片 resize 到目标分辨率

    Args:
        images: 原始图片列表（PIL Image 对象）
        text_tokens: 文本 token 数
        compression_ratios: 压缩比列表，如果为None则使用全局变量 COMPRESSION_RATIOS

    Returns:
        字典，key 为压缩比，value 为 (resized_images, target_resolution) 元组
    """
    if compression_ratios is None:
        compression_ratios = COMPRESSION_RATIOS

    resolution_list = [int(112 * m) for m in ([0.125, 0.25, 0.5] + list(range(1, 41)))]

    num_images = len(images)
    results = {}

    sorted_ratios = sorted(compression_ratios)

    for compression_ratio in sorted_ratios:
        image_token_limit = text_tokens / compression_ratio

        per_image_tokens = (
            image_token_limit / num_images if num_images > 0 else image_token_limit
        )

        theoretical_resolution = int(math.sqrt(per_image_tokens * 1024))
        theoretical_resolution = (theoretical_resolution // 16) * 16
        theoretical_resolution = max(224, min(4480, theoretical_resolution))

        previous_resolutions = [r[1] for r in results.values()] if results else []

        available_resolutions = resolution_list
        # Allow reusing resolutions to avoid forcing larger resolutions when small ones are taken
        # if previous_resolutions:
        #     available_resolutions = [
        #         r for r in resolution_list if r not in previous_resolutions
        #     ]
        #     if not available_resolutions:
        #         available_resolutions = resolution_list

        target_resolution = find_closest_resolution_prefer_larger(
            per_image_tokens, available_resolutions, tolerance_ratio=1.4
        )

        if compression_ratio <= 1.5:
            larger_than_theoretical = [
                r for r in available_resolutions if r > theoretical_resolution
            ]
            if larger_than_theoretical:
                larger_resolution = min(larger_than_theoretical)
                larger_tokens = calculate_image_tokens_qwen3(
                    larger_resolution, larger_resolution
                )
                if larger_tokens <= per_image_tokens * 1.4:
                    target_resolution = larger_resolution

        if theoretical_resolution in available_resolutions:
            theoretical_tokens = calculate_image_tokens_qwen3(
                theoretical_resolution, theoretical_resolution
            )
            closest_tokens = calculate_image_tokens_qwen3(
                target_resolution, target_resolution
            )
            if compression_ratio > 1.5:
                if abs(theoretical_tokens - per_image_tokens) < abs(
                    closest_tokens - per_image_tokens
                ):
                    target_resolution = theoretical_resolution
            elif theoretical_resolution > target_resolution:
                if abs(theoretical_tokens - per_image_tokens) < abs(
                    closest_tokens - per_image_tokens
                ):
                    target_resolution = theoretical_resolution

        if len(results) > 0:
            if target_resolution in previous_resolutions:
                max_prev_resolution = max(previous_resolutions)
                min_prev_resolution = min(previous_resolutions)

                if compression_ratio < 2:
                    larger_resolutions = [
                        r for r in available_resolutions if r > max_prev_resolution
                    ]
                    if larger_resolutions:
                        target_resolution = larger_resolutions[0]
                elif compression_ratio == 2:
                    larger_resolutions = [
                        r for r in available_resolutions if r > max_prev_resolution
                    ]
                    if larger_resolutions:
                        target_resolution = larger_resolutions[0]
                    else:
                        smaller_resolutions = [
                            r for r in available_resolutions if r < min_prev_resolution
                        ]
                        if smaller_resolutions:
                            target_resolution = smaller_resolutions[-1]
                else:
                    smaller_resolutions = [
                        r for r in available_resolutions if r < min_prev_resolution
                    ]
                    if smaller_resolutions:
                        target_resolution = smaller_resolutions[-1]

            for prev_ratio, (_, prev_resolution) in results.items():
                if compression_ratio < prev_ratio:
                    if target_resolution <= prev_resolution:
                        larger_resolutions = [
                            r for r in available_resolutions if r > prev_resolution
                        ]
                        if larger_resolutions:
                            target_resolution = larger_resolutions[0]
                elif compression_ratio > prev_ratio:
                    if target_resolution >= prev_resolution:
                        smaller_resolutions = [
                            r for r in available_resolutions if r < prev_resolution
                        ]
                        if smaller_resolutions:
                            target_resolution = smaller_resolutions[-1]

        resized_images = []
        for img in images:
            resized_img = img.resize(
                (target_resolution, target_resolution), PIL_Image.Resampling.LANCZOS
            )
            resized_images.append(resized_img)

        results[compression_ratio] = (resized_images, target_resolution)
        actual_tokens = calculate_image_tokens_qwen3(
            target_resolution, target_resolution
        )
        print(
            f"  压缩比 {compression_ratio}: 目标分辨率 {target_resolution}x{target_resolution}, "
            f"每张图片约 {actual_tokens} tokens (目标: {per_image_tokens:.1f} tokens)"
        )

    return results


def estimate_initial_font_size(text_tokens: int, resolution: int, line_height: float = 1.0) -> int:
    """
    基于经验公式快速估算初始字体大小。
    
    经验公式基于观察：在 monospace 字体下，平均每个字符宽度约等于字体大小的0.6倍，
    每个 token 平均约 3-4 个字符（考虑空格和标点）。
    可用区域 = (resolution - 2*margin) * (resolution - 2*margin)
    margin 约为 1% of resolution
    
    Args:
        text_tokens: 文本 token 数
        resolution: 分辨率（正方形）
        line_height: 行高倍数
        
    Returns:
        估算的字体大小
    """
    # margin 约为 1% of resolution
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    available_area = available_width * available_height
    
    # 估算：每个 token 约 3-4 个字符
    avg_chars_per_token = 3.5
    total_chars = text_tokens * avg_chars_per_token
    
    if total_chars <= 0:
        return 40  # 默认值
    
    # Monospace字体：字符宽度 ≈ font_size * 0.6，字符高度 ≈ font_size * line_height
    # 每个字符占用面积 ≈ (font_size * 0.6) * (font_size * line_height)
    # 总面积需求 ≈ total_chars * (font_size * 0.6) * (font_size * line_height)
    # 所以：font_size^2 ≈ available_area / (total_chars * 0.6 * line_height)
    
    # 考虑换行和布局因素，添加一个填充系数
    # 经验值：实际利用率约95%（尽量填满）
    fill_factor = 0.95
    
    estimated_fs_squared = (available_area * fill_factor) / (total_chars * 0.6 * line_height)
    estimated_fs = int(math.sqrt(estimated_fs_squared))
    
    # 倾向于选择稍大一点的字体（提高可读性）
    # 因为我们后续会检查更大的字体，所以这里可以稍微保守一点
    estimated_fs = int(estimated_fs * 1.1)  # 放大10%
    
    # 限制在合理范围内
    estimated_fs = max(4, min(150, estimated_fs))
    
    return estimated_fs


def estimate_page_count(
    text_tokens: int,
    resolution: int,
    font_size: int,
    line_height: float = 1.0,
) -> int:
    """
    快速估算给定配置下需要多少张图片（不实际渲染）。
    
    这个函数用于在 optimize_layout_config 中进行快速二分查找，
    避免频繁调用实际渲染函数。
    
    Args:
        text_tokens: 文本 token 数
        resolution: 分辨率（正方形）
        font_size: 字体大小
        line_height: 行高倍数
        
    Returns:
        估算的页数
    """
    # margin 约为 1% of resolution
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    if font_size <= 0 or available_width <= 0 or available_height <= 0:
        return 999  # 返回大值表示不可行
    
    # Monospace 字体特性：
    # 字符宽度 ≈ font_size * 0.6
    # 行高 ≈ font_size * line_height
    char_width = font_size * 0.6
    line_height_px = font_size * line_height
    
    # 每行能容纳的字符数
    chars_per_line = int(available_width / char_width) if char_width > 0 else 1
    chars_per_line = max(1, chars_per_line)
    
    # 每页能容纳的行数
    lines_per_page = int(available_height / line_height_px) if line_height_px > 0 else 1
    lines_per_page = max(1, lines_per_page)
    
    # 每页能容纳的字符数
    chars_per_page = chars_per_line * lines_per_page
    
    # 估算总字符数（每个 token 约 3.5 个字符）
    avg_chars_per_token = 3.5
    total_chars = text_tokens * avg_chars_per_token
    
    # 考虑换行符带来的额外行（代码中换行较多）
    # 估算：每 50 个字符约有 1 个换行
    estimated_newlines = total_chars / 50
    # 换行会导致行尾浪费，约浪费 30% 的字符空间
    effective_chars = total_chars * 1.3 + estimated_newlines * (chars_per_line * 0.3)
    
    # 计算需要的页数
    pages_needed = math.ceil(effective_chars / chars_per_page)
    pages_needed = max(1, pages_needed)
    
    return pages_needed


def estimate_fill_rate(
    text_tokens: int,
    resolution: int,
    font_size: int,
    line_height: float = 1.0,
    avg_line_length: int = 80,
) -> float:
    """
    估算文本在给定配置下的充满率。
    
    充满率 = 平均行宽 / 图片可用宽度
    
    对于代码文本，行长度通常在 40-120 字符之间，平均约 60-80 字符。
    如果图片宽度能容纳 200+ 字符，那么右边会有大量空白，充满率低。
    
    Args:
        text_tokens: 文本 token 数
        resolution: 分辨率（正方形）
        font_size: 字体大小
        line_height: 行高倍数
        avg_line_length: 代码平均行长度（字符数），默认 80
        
    Returns:
        充满率 (0.0 - 1.0+)，越接近 1.0 越好
    """
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    
    if font_size <= 0 or available_width <= 0:
        return 0.0
    
    # Monospace 字符宽度 ≈ font_size * 0.6
    char_width = font_size * 0.6
    
    # 每行能容纳的字符数
    chars_per_line = available_width / char_width if char_width > 0 else 1
    
    # 充满率 = 平均行长度 / 每行能容纳的字符数
    fill_rate = avg_line_length / chars_per_line if chars_per_line > 0 else 0
    
    # 限制在 0-1.5 范围内
    fill_rate = max(0.0, min(1.5, fill_rate))
    
    return fill_rate


def estimate_fill_rate_for_target_pages(
    text_tokens: int,
    resolution: int,
    target_pages: int,
    line_height: float = 1.0,
    avg_line_length: int = 80,
) -> float:
    """
    估算为了适应目标页数，实际的充满率会是多少。
    
    这个函数首先估算需要多大的字体才能产生 target_pages 页，
    然后基于该字体计算充满率。
    
    Args:
        text_tokens: 文本 token 数
        resolution: 分辨率（正方形）
        target_pages: 目标页数
        line_height: 行高倍数
        avg_line_length: 代码平均行长度
        
    Returns:
        估算的充满率
    """
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    if available_width <= 0 or available_height <= 0 or target_pages <= 0:
        return 0.0
    
    # 估算总字符数
    total_chars = text_tokens * 3.5
    
    # 每页需要容纳的字符数
    chars_per_page = total_chars / target_pages
    
    # 每页面积
    page_area = available_width * available_height
    
    # 每个字符占用面积 = char_width * line_height_px = (fs * 0.6) * (fs * line_height)
    # chars_per_page = page_area / char_area
    # char_area = page_area / chars_per_page
    # (fs * 0.6) * (fs * line_height) = page_area / chars_per_page
    # fs^2 = page_area / (chars_per_page * 0.6 * line_height)
    
    if chars_per_page <= 0:
        return 1.5  # 字符很少，肯定能填满
    
    fs_squared = page_area / (chars_per_page * 0.6 * line_height)
    estimated_fs = math.sqrt(fs_squared) if fs_squared > 0 else 4
    estimated_fs = max(4, min(150, estimated_fs))
    
    # 基于估算的字体计算充满率
    char_width = estimated_fs * 0.6
    chars_per_line = available_width / char_width if char_width > 0 else 1
    fill_rate = avg_line_length / chars_per_line if chars_per_line > 0 else 0
    
    return max(0.0, min(1.5, fill_rate))


def is_token_in_range(
    estimated_pages: int,
    per_image_tokens: int,
    target_tokens: float,
    min_ratio: float = 0.9,
    max_ratio: float = 1.1,
) -> bool:
    """
    检查实际 token 是否在目标范围内。
    
    Args:
        estimated_pages: 估算页数
        per_image_tokens: 每张图片的 token 数
        target_tokens: 目标 token 数
        min_ratio: 最小比例（默认 0.9）
        max_ratio: 最大比例（默认 1.1）
        
    Returns:
        是否在范围内
    """
    if target_tokens <= 0:
        return False
    
    actual_tokens = estimated_pages * per_image_tokens
    ratio = actual_tokens / target_tokens
    
    return min_ratio <= ratio <= max_ratio


def get_token_ratio(
    estimated_pages: int,
    per_image_tokens: int,
    target_tokens: float,
) -> float:
    """计算 token 比例"""
    if target_tokens <= 0:
        return 999.0
    actual_tokens = estimated_pages * per_image_tokens
    return actual_tokens / target_tokens


def analyze_text_structure(text: str) -> dict:
    """
    分析文本结构，提取行数、最长行、平均行长度等信息。
    
    Args:
        text: 原始文本
        
    Returns:
        dict: {
            'num_lines': 总行数,
            'max_line_chars': 最长行字符数,
            'avg_line_chars': 平均行字符数,
            'total_chars': 总字符数,
        }
    """
    # 处理制表符
    text = text.replace('\t', '    ')
    
    lines = text.split('\n')
    num_lines = len(lines)
    
    if num_lines == 0:
        return {
            'num_lines': 1,
            'max_line_chars': 1,
            'avg_line_chars': 1,
            'total_chars': 1,
        }
    
    line_lengths = [len(line) for line in lines]
    max_line_chars = max(line_lengths) if line_lengths else 1
    avg_line_chars = sum(line_lengths) / len(line_lengths) if line_lengths else 1
    total_chars = sum(line_lengths)
    
    # 确保最小值
    max_line_chars = max(1, max_line_chars)
    avg_line_chars = max(1, avg_line_chars)
    
    return {
        'num_lines': num_lines,
        'max_line_chars': max_line_chars,
        'avg_line_chars': avg_line_chars,
        'total_chars': total_chars,
    }


def calculate_optimal_font_size(
    resolution: int,
    pages: int,
    num_lines: int,
    max_line_chars: int,
    line_height: float = 1.0,
) -> int:
    """
    根据分辨率、张数、行数、最长行计算最优字体大小。
    
    字体大小的约束：
    - 高度约束：font_size * line_height * num_lines <= resolution * pages（所有行能放下）
    - 宽度约束：font_size * 0.6 * max_line_chars <= resolution（最长行不换行）
    
    最优字体 = min(高度上限, 宽度上限)
    
    Args:
        resolution: 分辨率（正方形）
        pages: 张数
        num_lines: 总行数
        max_line_chars: 最长行字符数
        line_height: 行高倍数
        
    Returns:
        最优字体大小
    """
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    # 高度约束：所有行能放下
    # font_size * line_height * num_lines <= available_height * pages
    # font_size <= (available_height * pages) / (num_lines * line_height)
    fs_height_limit = (available_height * pages) / (num_lines * line_height) if num_lines > 0 else 150
    
    # 宽度约束：最长行不换行
    # font_size * 0.6 * max_line_chars <= available_width
    # font_size <= available_width / (max_line_chars * 0.6)
    fs_width_limit = available_width / (max_line_chars * 0.6) if max_line_chars > 0 else 150
    
    # 取两个约束的最小值
    optimal_fs = min(fs_height_limit, fs_width_limit)
    optimal_fs = max(4, min(150, int(optimal_fs)))
    
    return optimal_fs


def calculate_fill_rate(
    font_size: int,
    resolution: int,
    pages: int,
    num_lines: int,
    avg_line_chars: int,
    line_height: float = 1.0,
) -> float:
    """
    计算充满率。
    
    垂直充满率 = font_size * line_height * num_lines / (available_height * pages)
    水平充满率 = font_size * 0.6 * avg_line_chars / available_width
    总充满率 = min(垂直, 水平)
    
    Args:
        font_size: 字体大小
        resolution: 分辨率
        pages: 张数
        num_lines: 总行数
        avg_line_chars: 平均行字符数
        line_height: 行高倍数
        
    Returns:
        充满率 (0.0 - 1.5)
    """
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    # 垂直充满率
    total_text_height = font_size * line_height * num_lines
    total_available_height = available_height * pages
    vertical_fill = total_text_height / total_available_height if total_available_height > 0 else 0
    
    # 水平充满率
    avg_line_width = font_size * 0.6 * avg_line_chars
    horizontal_fill = avg_line_width / available_width if available_width > 0 else 0
    
    # 取最小值作为总充满率
    fill_rate = min(vertical_fill, horizontal_fill)
    
    return min(1.5, fill_rate)


def optimize_layout_config(
    target_tokens: float,
    renderer_callback: Callable[[int, int, int], List[PIL_Image.Image]],
    previous_configs: List[Tuple[int, int]] = None,
    text_tokens: int = None,
    line_height: float = 1.0,
    text_structure: dict = None,
    compression_ratio: float = None,
    page_limit: int = 100,
) -> Tuple[List[PIL_Image.Image], int, int]:
    """
    寻找最佳的布局配置（分辨率、字体大小、张数），以适应目标 token 数。
    
    核心逻辑：
    1. 根据 target_tokens 的动态容差范围，计算每个分辨率对应的有效张数
    2. 对每个 (分辨率, 张数) 配置，根据文本行数和分辨率计算最优字体大小
    3. 数学计算充满率
    4. 使用综合评分系统（token 匹配度 + 充满率 + 分辨率 + 压缩率）选择最佳配置
    5. 只对最终选定的配置进行实际渲染
    
    Args:
        target_tokens: 目标总 token 数
        renderer_callback: 回调函数，接受 (width, height, font_size) 返回图片列表
        previous_configs: 之前已使用的配置列表 (resolution, image_count)，避免重复
        text_tokens: 原始文本 token 数
        line_height: 行高倍数（默认1.0）
        text_structure: 文本结构信息 {'num_lines', 'max_line_chars', 'avg_line_chars'}
        compression_ratio: 压缩比（用于调整分辨率权重）
        
    Returns:
        (best_images, best_resolution, best_font_size)
    """
    if previous_configs is None:
        previous_configs = []
    
    # 如果没有文本结构信息，使用默认估算
    if text_structure is None:
        # 从 text_tokens 估算
        estimated_chars = text_tokens * 3.5 if text_tokens else 10000
        text_structure = {
            'num_lines': int(estimated_chars / 60),  # 假设平均每行 60 字符
            'max_line_chars': 120,  # 假设最长行 120 字符
            'avg_line_chars': 60,   # 假设平均 60 字符
        }
    
    num_lines = text_structure['num_lines']
    max_line_chars = text_structure['max_line_chars']
    avg_line_chars = text_structure['avg_line_chars']
    
    # 分辨率列表：112 到 4480，步长 112
    resolutions = [112 * i for i in range(1, 41)]
    
    # 充满率阈值
    FILL_RATE_THRESHOLD = 0.90  # 90%
    
    # 动态调整容差范围
    if target_tokens < 50:
        token_min_ratio = 0.5
        token_max_ratio = 2.0
    elif target_tokens < 100:
        token_min_ratio = 0.7
        token_max_ratio = 1.5
    elif target_tokens < 3000:
        token_min_ratio = 0.8
        token_max_ratio = 1.25
    elif target_tokens < 5000:
        token_min_ratio = 0.9
        token_max_ratio = 1.12
    elif target_tokens < 10000:
        token_min_ratio = 0.93
        token_max_ratio = 1.08
    else:
        token_min_ratio = 0.95
        token_max_ratio = 1.05
    
    # ===== 第一步：构造所有有效的 (分辨率, 张数) 配置 =====
    all_configs = []
    
    for res in resolutions:
        per_image_tokens = calculate_image_tokens_qwen3(res, res)
        
        # 根据动态容差范围计算有效张数
        min_pages = math.ceil(target_tokens * token_min_ratio / per_image_tokens)
        max_pages = math.floor(target_tokens * token_max_ratio / per_image_tokens)
        
        # 筛选 pages >= 1
        min_pages = max(1, min_pages)
        
        if min_pages > max_pages:
            continue  # 这个分辨率没有有效的张数
        
        # 对于每个有效张数（动态调整最大页数限制）
        max_allowed_pages = page_limit
        for pages in range(min_pages, max_pages + 1):
            if pages > max_allowed_pages:
                continue

            # 检查是否是新配置
            is_new = (res, pages) not in previous_configs
            
            # ===== 第二步：计算最优字体大小 =====
            optimal_fs = calculate_optimal_font_size(
                res, pages, num_lines, max_line_chars, line_height
            )
            
            # ===== 第三步：计算充满率 =====
            fill_rate = calculate_fill_rate(
                optimal_fs, res, pages, num_lines, avg_line_chars, line_height
            )
            
            # 计算实际 token
            actual_tokens = pages * per_image_tokens
            token_ratio = actual_tokens / target_tokens if target_tokens > 0 else 1.0
            
            # 过滤规则（保留更多候选，极端情况才过滤）
            if target_tokens >= 10000:
                if fill_rate < 0.03 and optimal_fs < 6:
                    continue
            elif target_tokens >= 5000:
                if fill_rate < 0.1 and optimal_fs < 5:
                    continue
            elif target_tokens >= 3000:
                if fill_rate < 0.1 and optimal_fs < 5:
                    continue
            
            all_configs.append({
                'resolution': res,
                'pages': pages,
                'font_size': optimal_fs,
                'fill_rate': fill_rate,
                'tokens': actual_tokens,
                'token_ratio': token_ratio,
                'is_new': is_new,
                'per_image_tokens': per_image_tokens,
            })
    
    if not all_configs:
        # 没有有效配置，使用 fallback 策略: 112x112x1, font>=3
        print("Warning: No valid layout found for target tokens. Using fallback strategy (112x112x1).")
        res = 112
        
        # 二分查找：找到能放入1页的最大字体（最小为3）
        low = 3
        high = 150
        best_fs = 3
        
        # 先检查最小值是否可行（或者是否必须使用最小值）
        min_imgs = renderer_callback(res, res, 3)
        if len(min_imgs) > 1:
            print(f"  [Fallback] Text too long for 112x112x1 even at font 3. Using font 3 ({len(min_imgs)} pages).")
            min_imgs = min_imgs[:page_limit]
            return min_imgs, res, 3
            
        while low <= high:
            mid = (low + high) // 2
            imgs = renderer_callback(res, res, mid)
            if len(imgs) == 1:
                best_fs = mid
                low = mid + 1  # 尝试更大的字体
            else:
                high = mid - 1  # 字体太大，导致分页，需要减小
        
        imgs = renderer_callback(res, res, best_fs)
        print(f"  [Fallback] Using 112x112x1 with font {best_fs}")
        return imgs, res, best_fs
    
    # ===== 第四步：使用综合评分系统选择最佳配置 =====
    def calculate_config_score(c, compression_ratio=None):
        """
        计算配置的综合得分（越高越好）
        
        考虑因素：
        1. Token 匹配度：实际 tokens 与目标 tokens 的接近程度
        2. 充满率：90% 以上同等级，20% 以下严重惩罚
        3. 分辨率：在充满率相近时优先选大分辨率
        4. 压缩率：小压缩率时更看重分辨率
        """
        # 1. Token 匹配度评分 (0-1)
        token_diff = abs(c['token_ratio'] - 1.0)
        if target_tokens < 50:
            token_penalty_factor = 1.0
        elif target_tokens < 100:
            token_penalty_factor = 1.5
        elif target_tokens < 3000:
            token_penalty_factor = 2.0
        elif target_tokens < 5000:
            token_penalty_factor = 2.5
        elif target_tokens < 10000:
            token_penalty_factor = 3.0
        else:
            token_penalty_factor = 3.5
        
        token_score = 1.0 / (1.0 + token_diff * token_penalty_factor)
        
        # 2. 充满率评分 (0-1)
        fill_rate = c['fill_rate']
        
        # 对于小 token 场景，充满率权重大幅降低
        if target_tokens < 50 or (compression_ratio is not None and compression_ratio >= 4.0):
            # 小 token/高压缩率：充满率只要不是极低就可以（通过调字体填充）
            if fill_rate >= 0.2:
                fill_score = 0.9  # 20% 以上都视为可接受
            else:
                fill_score = 0.5  # 20% 以下轻微惩罚（而非严重惩罚）
        else:
            # 正常/大 token 场景：充满率非常重要
            if fill_rate >= 0.9:
                fill_score = 1.0  # 90% 以上视为同一等级
            elif fill_rate >= 0.2:
                # 20%-90% 之间线性映射到 0.2-1.0
                fill_score = 0.2 + (fill_rate - 0.2) * (0.8 / 0.7)
            elif fill_rate >= 0.1:
                # 10%-20% 之间：严重惩罚
                fill_score = 0.05
            else:
                # <10%：极端惩罚（几乎排除）
                fill_score = 0.01
        
        resolution_normalized = (c['resolution'] - 112) / (4480 - 112)
        resolution_bonus = 1.0 + resolution_normalized * (0.5 if target_tokens >= 3000 else 0.3)
        
        # 4. 压缩率对分辨率权重的影响
        if compression_ratio is not None and compression_ratio <= 2.0:
            # 小压缩率（0.5x, 1x, 1.5x, 2x）：提高分辨率权重
            compression_bonus = 1.2
        else:
            compression_bonus = 1.0
        
        # 5. 动态调整权重：小 token 或高压缩率时，token 匹配度更重要
        if target_tokens < 50 or (compression_ratio is not None and compression_ratio >= 4.0):
            # 小 token 或高压缩率场景：token 匹配度主导，fill_score 影响极小
            # 公式：token_score^2.5 * fill_score^0.2 （token 主导，fill 几乎不影响）
            score = (token_score ** 2.5) * (fill_score ** 0.2) * resolution_bonus * compression_bonus
        elif target_tokens < 100:
            score = (token_score ** 1.5) * (fill_score ** 0.75) * resolution_bonus * compression_bonus
        else:
            score = token_score * fill_score * resolution_bonus * compression_bonus
            if target_tokens >= 5000 and c.get('font_size', 10) < 8:
                score *= 0.8
        
        return score
    
    # 为所有配置计算综合得分（包括已使用的配置）
    for c in all_configs:
        c['score'] = calculate_config_score(c, compression_ratio=compression_ratio)
        # 对已使用的配置给予轻微惩罚（-5%），但不完全排除
        if not c['is_new']:
            c['score'] *= 0.95
    
    # 按得分排序（降序）
    all_configs.sort(key=lambda x: -x['score'])
    
    # 选择策略：选择得分最高的前 5 个进行实际渲染验证
    # 优先选择新配置，但如果旧配置得分明显更高也可以选
    selected = all_configs[:min(5, len(all_configs))]
    
    # 从 selected 中选择得分最高的作为初步最佳配置
    best = selected[0]
    
    # ===== 第五步：实际渲染 =====
    imgs = renderer_callback(best['resolution'], best['resolution'], best['font_size'])
    actual_pages = len(imgs)
    
    # 如果实际页数和预期不符，调整字体大小
    if actual_pages != best['pages']:
        # 二分查找合适的字体大小
        low_fs = 4
        high_fs = 150
        target_pages = best['pages']
        best_fs = best['font_size']
        best_imgs = imgs
        
        while low_fs <= high_fs:
            mid_fs = (low_fs + high_fs) // 2
            test_imgs = renderer_callback(best['resolution'], best['resolution'], mid_fs)
            
            if len(test_imgs) <= target_pages:
                best_fs = mid_fs
                best_imgs = test_imgs
                low_fs = mid_fs + 1
            else:
                high_fs = mid_fs - 1
        
        imgs = best_imgs
        best['font_size'] = best_fs
        actual_pages = len(imgs)
    
    # ===== 第六步：尝试更大的字体以提高可读性 =====
    # 即使页数符合预期，也应该尝试更大的字体（只要页数不增加）
    current_best_fs = best['font_size']
    current_best_imgs = imgs
    
    # 使用二分查找找到最大的可用字体
    # 这比线性尝试更高效，也能找到更大的字体
    low_fs = current_best_fs
    high_fs = 150
    target_pages = actual_pages
    
    # 先快速测试一个较大的字体，看看是否有提升空间
    test_fs = min(current_best_fs + 20, 150)
    test_imgs = renderer_callback(best['resolution'], best['resolution'], test_fs)
    if len(test_imgs) <= target_pages:
        # 有很大提升空间，使用二分查找
        low_fs = test_fs
        current_best_fs = test_fs
        current_best_imgs = test_imgs
    
    # 二分查找最大可用字体
    while low_fs < high_fs - 1:
        mid_fs = (low_fs + high_fs) // 2
        test_imgs = renderer_callback(best['resolution'], best['resolution'], mid_fs)
        
        if len(test_imgs) <= target_pages:
            current_best_fs = mid_fs
            current_best_imgs = test_imgs
            low_fs = mid_fs
        else:
            high_fs = mid_fs
    
    return current_best_imgs, best['resolution'], current_best_fs


def _optimize_layout_config_slow(
    target_tokens: float,
    renderer_callback: Callable[[int, int, int], List[PIL_Image.Image]],
    previous_configs: List[Tuple[int, int]],
    line_height: float = 1.0,
) -> Tuple[List[PIL_Image.Image], int, int]:
    """
    原始的慢速版本（当没有 text_tokens 时使用）。
    实际调用渲染函数进行搜索。
    """
    # 分辨率列表：只尝试几个关键分辨率以加速
    resolutions = [112 * i for i in [20, 16, 12, 8, 4, 2, 1]]
    
    min_fs = 4
    max_fs = 150
    
    strict_candidates = []
    relaxed_candidates = []

    for res in resolutions:
        per_image_tokens = calculate_image_tokens_qwen3(res, res)
        token_limit = target_tokens * 1.25
        
        imgs_min = renderer_callback(res, res, min_fs)
        min_needed = len(imgs_min)
        
        if min_needed == 0:
            continue
        
        current_tokens = min_needed * per_image_tokens
        
        if current_tokens <= token_limit:
            max_images_limit = int(token_limit // per_image_tokens)
            
            low = min_fs
            high = max_fs
            curr_best_fs = min_fs
            curr_best_imgs = imgs_min
            
            while low <= high:
                mid = (low + high) // 2
                if mid == low:
                    if high > low:
                        imgs_high = renderer_callback(res, res, high)
                        if len(imgs_high) <= max_images_limit:
                            curr_best_fs = high
                            curr_best_imgs = imgs_high
                    break
                
                imgs = renderer_callback(res, res, mid)
                if len(imgs) <= max_images_limit:
                    curr_best_fs = mid
                    curr_best_imgs = imgs
                    low = mid + 1
                else:
                    high = mid - 1
            
            is_new = (res, len(curr_best_imgs)) not in previous_configs
            score = res * 1000 + curr_best_fs
            strict_candidates.append((score, curr_best_fs, res, curr_best_imgs, is_new))
            
        else:
            limit_images = min_needed
            
            low = min_fs
            high = max_fs
            curr_best_fs = min_fs
            curr_best_imgs = imgs_min
            
            while low <= high:
                mid = (low + high) // 2
                if mid == low:
                    if high > low:
                        imgs_high = renderer_callback(res, res, high)
                        if len(imgs_high) <= limit_images:
                            curr_best_fs = high
                            curr_best_imgs = imgs_high
                    break
                
                imgs = renderer_callback(res, res, mid)
                if len(imgs) <= limit_images:
                    curr_best_fs = mid
                    curr_best_imgs = imgs
                    low = mid + 1
                else:
                    high = mid - 1
            
            is_new = (res, len(curr_best_imgs)) not in previous_configs
            total_tokens = len(curr_best_imgs) * per_image_tokens
            relaxed_candidates.append((total_tokens, curr_best_fs, res, curr_best_imgs, is_new))

    # 决策阶段
    strict_new = [c for c in strict_candidates if c[4]]
    if strict_new:
        strict_new.sort(key=lambda x: x[0], reverse=True)
        best = strict_new[0]
        return best[3], best[2], best[1]
        
    if strict_candidates:
        strict_candidates.sort(key=lambda x: x[0], reverse=True)
        best = strict_candidates[0]
        return best[3], best[2], best[1]
        
    if relaxed_candidates:
        relaxed_candidates.sort(key=lambda x: (x[0], -x[1], -x[2]))
        best_overall = relaxed_candidates[0]
        
        relaxed_new = [c for c in relaxed_candidates if c[4]]
        if relaxed_new:
            best_new = relaxed_new[0]
            if best_new[0] <= best_overall[0] * 1.1:
                return best_new[3], best_new[2], best_new[1]
            else:
                return best_overall[3], best_overall[2], best_overall[1]
        else:
            return best_overall[3], best_overall[2], best_overall[1]

    print("Warning: No valid layout found for target tokens. Using fallback strategy.")
    # Fallback: 112x112, 1 image, font size >= 3 (max possible)
    res = 112
    limit_images = 1
    
    # Binary search for max font size that fits in 1 image (or min 3)
    low = 3
    high = 150
    best_fs = 3
    
    # 先检查 font=3 是否能放下（或者至少这是我们能做的最好的）
    imgs_min = renderer_callback(res, res, 3)
    best_imgs = imgs_min[:1] # Force 1 image if needed, but renderer returns list
    
    # 如果 renderer 返回多张图，说明 font=3 也放不下全部文本。
    # 但根据要求 "112*112*1"，我们只能取第一张。
    # 并且 "字体最小值为3... 选择能充满画面的最大字体"
    # 如果 font=3 都超了，那只能 font=3。
    # 如果 font=3 没超（imgs_min 长度为 1），我们可以尝试变大字体。
    
    if len(imgs_min) <= 1:
        # 尝试寻找更大的字体
        while low <= high:
            mid = (low + high) // 2
            if mid == low:
                if high > low:
                    imgs_high = renderer_callback(res, res, high)
                    if len(imgs_high) <= 1:
                        best_fs = high
                        best_imgs = imgs_high
                break
            
            imgs = renderer_callback(res, res, mid)
            if len(imgs) <= 1:
                best_fs = mid
                best_imgs = imgs
                low = mid + 1
            else:
                high = mid - 1
    
    # 确保只返回1张图
    if len(best_imgs) > 1:
        best_imgs = best_imgs[:1]
        
    return best_imgs, res, best_fs


def generate_compressed_images_dynamic(
    text_tokens: int,
    renderer_func: Callable[[int, int, int], List[PIL_Image.Image]],
    compression_ratios: List[float] = None,
    text_structure: dict = None,
    data_id: str = None,
    page_limit: int = 100,
) -> Dict[float, Tuple[List[PIL_Image.Image], int, int]]:
    """
    根据压缩比动态生成图片，而非简单resize。
    
    Args:
        text_tokens: 文本 token 数
        renderer_func: 渲染函数
        compression_ratios: 压缩比列表
        text_structure: 文本结构信息 {'num_lines', 'max_line_chars', 'avg_line_chars'}
    
    Returns:
        Dict[ratio, (images, resolution, font_size)]
    """
    if compression_ratios is None:
        compression_ratios = COMPRESSION_RATIOS
    
    results = {}
    sorted_ratios = sorted(compression_ratios)
    resolution_list = [int(112 * m) for m in ([0.125, 0.25, 0.5] + list(range(1, 41)))]
    base_ratio = 1.0
    base_start = time.time()
    base_target_tokens = text_tokens / base_ratio
    base_imgs, base_res, base_fs = optimize_layout_config(
        base_target_tokens,
        renderer_func,
        previous_configs=[],
        text_tokens=text_tokens,
        line_height=1.0,
        text_structure=text_structure,
        compression_ratio=base_ratio,
        page_limit=page_limit,
    )
    base_elapsed = time.time() - base_start
    base_actual_tokens = len(base_imgs) * calculate_image_tokens_qwen3(base_res, base_res)
    if text_structure:
        base_fill = calculate_fill_rate(
            base_fs, base_res, len(base_imgs),
            text_structure['num_lines'],
            text_structure['avg_line_chars'],
            1.0
        )
    else:
        base_fill = estimate_fill_rate(text_tokens, base_res, base_fs, 1.0)
    base_id_prefix = f"[{data_id}] " if data_id else ""
    base_log = f"  {base_id_prefix}Ratio {base_ratio}: Res {base_res}x{base_res}, Count {len(base_imgs)}, Font {base_fs}, Fill {base_fill:.0%}, Tokens {base_actual_tokens} (Target {base_target_tokens:.1f}) [耗时: {base_elapsed:.3f}s]"
    base_warn = []
    base_diff = abs(base_actual_tokens - base_target_tokens) / base_target_tokens if base_target_tokens > 0 else 0
    if base_diff > 0.2:
        base_warn.append(f"⚠️ Token失衡: {base_diff:.1%}")
    if base_fill < 0.1:
        base_warn.append(f"⚠️ 充满率过低: {base_fill:.1%}")
    if base_fs < 8:
        base_warn.append(f"⚠️ 字体过小: {base_fs}")
    used_resolutions = set()
    prev_selected_res = None
    if base_ratio in sorted_ratios:
        results[base_ratio] = (base_imgs, base_res, base_fs)
        used_resolutions.add(base_res)
        prev_selected_res = base_res
        if base_warn:
            print(f"{base_log} {' '.join(base_warn)}")
        else:
            print(base_log)
    for ratio in sorted_ratios:
        if ratio == base_ratio:
            continue
        r_start = time.time()
        target_tokens = text_tokens / ratio
        num_images = len(base_imgs)
        per_img_target = target_tokens / num_images if num_images > 0 else target_tokens
        target_res = find_closest_resolution_prefer_larger(per_img_target, resolution_list, tolerance_ratio=1.4)
        selected_res = target_res
        if selected_res in used_resolutions or (prev_selected_res is not None and selected_res > prev_selected_res):
            not_used = [r for r in resolution_list if r not in used_resolutions]
            smaller_ok = [r for r in not_used if prev_selected_res is None or r <= prev_selected_res]
            if smaller_ok:
                le_target = [r for r in smaller_ok if r <= target_res]
                selected_res = max(le_target) if le_target else max(smaller_ok)
            elif not_used:
                selected_res = min(not_used, key=lambda r: abs(calculate_image_tokens_qwen3(r, r) - per_img_target))
        used_resolutions.add(selected_res)
        prev_selected_res = selected_res
        resized = []
        for img in base_imgs:
            resized.append(img.resize((selected_res, selected_res), PIL_Image.Resampling.LANCZOS))
        actual_tokens = num_images * calculate_image_tokens_qwen3(selected_res, selected_res)
        if text_structure:
            fill_rate = calculate_fill_rate(
                base_fs, selected_res, num_images,
                text_structure['num_lines'],
                text_structure['avg_line_chars'],
                1.0
            )
        else:
            fill_rate = estimate_fill_rate(text_tokens, selected_res, base_fs, 1.0)
        results[ratio] = (resized, selected_res, base_fs)
        r_elapsed = time.time() - r_start
        id_prefix = f"[{data_id}] " if data_id else ""
        log = f"  {id_prefix}Ratio {ratio}: Res {selected_res}x{selected_res}, Count {num_images}, Font {base_fs}, Fill {fill_rate:.0%}, Tokens {actual_tokens} (Target {target_tokens:.1f}) [耗时: {r_elapsed:.3f}s]"
        warns = []
        diff_ratio = abs(actual_tokens - target_tokens) / target_tokens if target_tokens > 0 else 0
        if diff_ratio > 0.2:
            warns.append(f"⚠️ Token失衡: {diff_ratio:.1%}")
        if fill_rate < 0.1:
            warns.append(f"⚠️ 充满率过低: {fill_rate:.1%}")
        if base_fs < 8:
            warns.append(f"⚠️ 字体过小: {base_fs}")
        if warns:
            print(f"{log} {' '.join(warns)}")
        else:
            print(log)
    return results


def calculate_image_tokens_from_paths(image_paths: List[str]) -> int:
    """根据图片路径计算图片的token数量（使用patch估算方法）"""
    total_tokens = 0

    BASE_IMAGE_TOKENS = 170
    PATCH_SIZE = 14

    for image_path in image_paths:
        try:
            with PIL_Image.open(image_path) as img:
                width, height = img.size

            num_patches_w = math.ceil(width / PATCH_SIZE)
            num_patches_h = math.ceil(height / PATCH_SIZE)
            total_patches = num_patches_w * num_patches_h

            image_tokens = BASE_IMAGE_TOKENS + min(total_patches * 2, 100)
            total_tokens += int(image_tokens)
        except Exception as e:
            print(f"  警告: 计算图片 {image_path} 的token时出错: {e}，使用默认值")
            total_tokens += BASE_IMAGE_TOKENS

    return total_tokens


def calculate_image_tokens_with_processor(
    image_paths: List[str], processor: Optional[AutoProcessor] = None
) -> Optional[int]:
    """使用 AutoProcessor 计算图片的token数量"""
    if not TRANSFORMERS_AVAILABLE:
        return None

    if processor is None:
        try:
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-235B-A22B-Instruct", trust_remote_code=True
            )
        except Exception as e:
            print(f"  警告: AutoProcessor 加载失败，尝试 force_download=True...: {e}")
            try:
                processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    trust_remote_code=True,
                    force_download=True,
                )
            except Exception as e2:
                print(f"  警告: 加载 AutoProcessor 再次失败: {e2}")
                return None

    total_tokens = 0

    for image_path in image_paths:
        try:
            image = PIL_Image.open(image_path).convert("RGB")

            # 构造消息（只包含图片，不包含文本）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": ""},  # 空文本，只计算图片tokens
                    ],
                }
            ]

            # 使用 processor 处理
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # 从 inputs 中提取 token 信息
            image_tokens = 0

            # 方法1: 使用 image_grid_thw 计算（最准确）
            if "image_grid_thw" in inputs:
                grid_info = inputs["image_grid_thw"]
                # grid_info shape: [1, 3] -> [num_images, height_grid, width_grid]
                num_images = grid_info[0][0].item()
                height_grid = grid_info[0][1].item()
                width_grid = grid_info[0][2].item()
                image_tokens = height_grid * width_grid
            # 方法2: 使用 pixel_values 的第一维
            elif "pixel_values" in inputs:
                image_tokens = inputs["pixel_values"].shape[0]
            else:
                # 备用方法：使用patch估算
                image_tokens = calculate_image_tokens_from_paths([image_path])

            total_tokens += image_tokens

        except Exception as e:
            print(f"  警告: 使用Processor计算图片 {image_path} 的token时出错: {e}")
            # 使用默认估算值
            total_tokens += calculate_image_tokens_from_paths([image_path])

    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="紧凑型图片生成工具")
    parser.add_argument("--filename", type=str, default=None, help="原始文件名")
    parser.add_argument("--txt-file", type=str, default=None, help="txt文件")
    parser.add_argument(
        "--output-dir", type=str, default="./generated_images_compact", help="输出目录"
    )
    parser.add_argument("--width", type=int, default=2240, help="宽度 (默认2240)")
    parser.add_argument("--height", type=int, default=2240, help="高度 (默认2240)")
    parser.add_argument("--font-size", type=int, default=40, help="字体大小 (默认40)")
    parser.add_argument("--line-height", type=float, default=1.0, help="行高 (默认1.0)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI")
    parser.add_argument(
        "--preserve-newlines",
        action="store_true",
        default=True,
        help="保留换行符 (默认True)",
    )
    parser.add_argument(
        "--enable-syntax-highlight", action="store_true", help="语法高亮"
    )
    parser.add_argument("--crop-whitespace", action="store_true", help="裁剪白边")
    parser.add_argument("--enable-two-column", action="store_true", help="双栏")
    parser.add_argument(
        "--resize-mode", action="store_true", default=True, help="resize模式 (默认True)"
    )
    parser.add_argument(
        "--no-resize-mode",
        action="store_false",
        dest="resize_mode",
        help="禁用resize模式",
    )
    parser.add_argument("--enable-bold", action="store_true", help="加粗")

    args = parser.parse_args()

    if args.txt_file:
        with open(args.txt_file, "r") as f:
            source_code = f.read()
        filename = os.path.basename(args.txt_file)
    elif args.filename:
        # Try to load from repoqa or just read file
        if os.path.exists(args.filename):
            with open(args.filename, "r") as f:
                source_code = f.read()
            filename = args.filename
        else:
            # If not existing locally, maybe just use it as name (repoqa style)
            # But here we only do image generation, so we need content.
            print(
                f"File not found: {args.filename}. For RepoQA files, use run_pipeline.py"
            )
            return
    else:
        print("Please provide --filename or --txt-file")
        return

    generate_images_for_file(
        filename,
        source_code,
        args.output_dir,
        args.width,
        args.height,
        args.font_size,
        args.line_height,
        args.dpi,
        preserve_newlines=args.preserve_newlines,
        enable_syntax_highlight=args.enable_syntax_highlight,
        should_crop_whitespace=args.crop_whitespace,
        enable_two_column=args.enable_two_column,
        enable_bold=args.enable_bold,
    )


if __name__ == "__main__":
    main()
