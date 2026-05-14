from llm_response_analyzer.readability_features import get_flesch_reading_ease, get_flesch_kincaid_grade


class TestGetFleschReadingEase:
    def test_flesch_reading_ease_empty_text(self):
        text = ""
        result = get_flesch_reading_ease(text, "en")

        assert result == 0

    def test_flesch_reading_ease_single_word(self):
        text = "hello"
        result = get_flesch_reading_ease(text, "en")

        assert isinstance(result, (int, float))

    def test_flesch_reading_ease_very_simple_text(self):
        text = "I like cats. Cats are fun. I play with cats."
        result = get_flesch_reading_ease(text, "en")

        assert result > 80  # Very easy to read

    def test_flesch_reading_ease_simple_text(self):
        text = "The cat sat on the mat. It was a sunny day. The cat was happy."
        result = get_flesch_reading_ease(text, "en")

        assert result > 70

    def test_flesch_reading_ease_complex_text(self):
        text = "The multifaceted ramifications of sophisticated technological advancements necessitate comprehensive reevaluation of contemporary pedagogical methodologies."
        result = get_flesch_reading_ease(text, "en")

        assert result < 50

    def test_flesch_reading_ease_professional_text(self):
        text = "Notwithstanding the aforementioned considerations, the paradigmatic framework elucidates multifarious hermeneutical ambiguities inherent within phenomenological exegesis."
        result = get_flesch_reading_ease(text, "en")

        assert result < 30

    def test_flesch_reading_ease_return_type(self):
        text = "This is a test sentence for readability analysis."
        result = get_flesch_reading_ease(text, "en")

        assert isinstance(result, (int, float))

    def test_flesch_reading_ease_multiple_sentences(self):
        text = "I love reading books. Books are fun. Reading helps me learn."
        result = get_flesch_reading_ease(text, "en")

        assert result > 60

    def test_flesch_reading_ease_with_punctuation(self):
        text = "Hello, world! How are you? I am fine!!!"
        result = get_flesch_reading_ease(text, "en")

        assert isinstance(result, (int, float))

    def test_flesch_reading_ease_question_marks(self):
        text = "Do you like cats? What about dogs? Do you prefer both?"
        result = get_flesch_reading_ease(text, "en")

        assert result > 50

    def test_flesch_reading_ease_english_language(self):
        text = "The quick brown fox jumps over the lazy dog."
        result_en = get_flesch_reading_ease(text, "en")

        assert result_en > 0

    def test_flesch_reading_ease_long_text(self):
        text = (
            "Python is a popular programming language. It is used for web development. "
            "It is also used for data science. Many developers prefer Python. Python is easy to learn."
        )
        result = get_flesch_reading_ease(text, "en")

        assert result > 40


class TestGetFleschKincaidGrade:
    def test_flesch_kincaid_grade_empty_text(self):
        text = ""
        result = get_flesch_kincaid_grade(text, "en")

        assert result == 0

    def test_flesch_kincaid_grade_single_word(self):
        text = "hello"
        result = get_flesch_kincaid_grade(text, "en")

        assert isinstance(result, (int, float))

    def test_flesch_kincaid_grade_very_simple_text(self):
        text = "I like cats. Cats are fun. I play with cats."
        result = get_flesch_kincaid_grade(text, "en")

        assert result < 6  # Below elementary school level

    def test_flesch_kincaid_grade_simple_text(self):
        text = "The cat sat on the mat. It was a sunny day. The cat was happy."
        result = get_flesch_kincaid_grade(text, "en")

        assert result < 8  # Elementary to early middle school

    def test_flesch_kincaid_grade_moderate_text(self):
        text = (
            "Environmental conservation requires comprehensive understanding of ecological systems. "
            "Sustainable practices promote long-term environmental health."
        )
        result = get_flesch_kincaid_grade(text, "en")

        assert result > 8  # At least middle school level

    def test_flesch_kincaid_grade_complex_text(self):
        text = (
            "The multifaceted ramifications of sophisticated technological advancements necessitate "
            "comprehensive reevaluation of contemporary pedagogical methodologies and epistemological frameworks."
        )
        result = get_flesch_kincaid_grade(text, "en")

        assert result > 12

    def test_flesch_kincaid_grade_return_type(self):
        text = "This is a test sentence for readability analysis."
        result = get_flesch_kincaid_grade(text, "en")

        assert isinstance(result, (int, float))

    def test_flesch_kincaid_grade_multiple_sentences(self):
        text = "I love reading books because they are fun. Reading helps me learn and focus. My favorite books are science fiction and fantasy."
        result = get_flesch_kincaid_grade(text, "en")

        assert result > 0

    def test_flesch_kincaid_grade_with_punctuation(self):
        text = "Hello, world! How are you? I am fine!!!"
        result = get_flesch_kincaid_grade(text, "en")

        assert isinstance(result, (int, float))

    def test_flesch_kincaid_grade_question_marks(self):
        text = "Do you prefer cats or dogs? What is you favourite breed?"
        result = get_flesch_kincaid_grade(text, "en")

        assert result > 0

    def test_flesch_kincaid_grade_english_language(self):
        text = "The quick brown fox jumps over the lazy dog."
        result_en = get_flesch_kincaid_grade(text, "en")

        assert result_en > 0

    def test_flesch_kincaid_grade_long_text(self):
        text = (
            "Python is a popular programming language. It is used for web development. "
            "It is also used for data science. Many developers prefer Python. Python is easy to learn."
        )
        result = get_flesch_kincaid_grade(text, "en")

        assert result > 0

    def test_flesch_kincaid_grade_technical_text(self):
        text = (
            "Machine learning algorithms utilize statistical methodologies to facilitate "
            "pattern recognition within multidimensional datasets through iterative optimization procedures."
        )
        result = get_flesch_kincaid_grade(text, "en")

        assert result > 13


class TestReadabilityComparison:
    def test_simple_vs_complex_text_fre(self):
        simple_text = "I like cats. Cats are fun."
        complex_text = "The multifaceted ramifications of technological advancements necessitate reevaluation."

        simple_fre = get_flesch_reading_ease(simple_text, "en")
        complex_fre = get_flesch_reading_ease(complex_text, "en")

        assert simple_fre > complex_fre

    def test_simple_vs_complex_text_grade(self):
        simple_text = "I like cats. Cats are fun."
        complex_text = "The multifaceted ramifications of technological advancements necessitate reevaluation."

        simple_grade = get_flesch_kincaid_grade(simple_text, "en")
        complex_grade = get_flesch_kincaid_grade(complex_text, "en")

        assert simple_grade < complex_grade

    def test_fre_and_grade_correlation(self):
        text = "The quick brown fox jumps over the lazy dog. This is a test."
        fre = get_flesch_reading_ease(text, "en")
        grade = get_flesch_kincaid_grade(text, "en")

        if fre > 80:
            assert grade < 8
        elif fre < 50:
            assert grade > 12

    def test_readability_consistency(self):
        text = "Python is a popular programming language used worldwide."
        fre1 = get_flesch_reading_ease(text, "en")
        fre2 = get_flesch_reading_ease(text, "en")
        grade1 = get_flesch_kincaid_grade(text, "en")
        grade2 = get_flesch_kincaid_grade(text, "en")

        assert fre1 == fre2
        assert grade1 == grade2

    def test_very_short_vs_very_long_text(self):
        short_text = "I like cats."
        long_text = " ".join(["I like cats."] * 10)

        short_fre = get_flesch_reading_ease(short_text, "en")
        long_fre = get_flesch_reading_ease(long_text, "en")

        assert short_fre > 70 or long_fre > 70
