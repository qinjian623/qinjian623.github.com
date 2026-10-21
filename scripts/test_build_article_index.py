import unittest

from build_article_index import choose_topic, clean_excerpt


class ImageDescriptionTests(unittest.TestCase):
    def test_image_descriptions_do_not_change_topics_or_excerpts(self):
        original = "# 节日快乐\n老模型，新数据。\n![](sample.jpg)"
        described = original.replace("![]", "![道路图像上的目标检测结果]")
        self.assertEqual(
            choose_topic("节日快乐", "bit_ass/2018-10-24-节日快乐.md", original),
            choose_topic("节日快乐", "bit_ass/2018-10-24-节日快乐.md", described),
        )
        self.assertEqual(
            clean_excerpt(original, "节日快乐"),
            clean_excerpt(described, "节日快乐"),
        )
