import textstat


def get_flesch_reading_ease(text: str, language: str = "en"):
    if not text:
        return 0

    textstat.set_lang(language)
    fre = textstat.flesch_reading_ease(text)

    return fre


def get_flesch_kincaid_grade(text: str, language: str = "en"):
    if not text:
        return 0

    textstat.set_lang(language)
    fkgl = textstat.flesch_kincaid_grade(text)

    return fkgl
