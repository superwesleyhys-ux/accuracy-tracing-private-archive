"""Deterministic, immutable target obligations for staged verification.

The plan is derived only from the frozen target text.  It never sees material,
model output or gold labels, so later retrieval cannot silently rewrite what
has to be checked.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import re


TARGET_PLAN_VERSION = "deterministic-target-plan-v4"
MAX_PROBES = 8

_SENTENCE_BREAK = re.compile(r"[;；。!?！？]+|(?<!\d)\.(?!\d)")
_CLAUSE_BREAK = re.compile(r"\s+(?:and|as well as)\s+|并且|且", re.IGNORECASE)
_CONDITIONAL = re.compile(r"\b(?:if|unless|provided that|when)\b|如果|若|除非", re.IGNORECASE)
_MONTHS = ("january|february|march|april|may|june|july|august|"
           "september|october|november|december")
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_YEAR = r"(?:19|20)\d{2}"
_QUARTER_PERIOD = (
    rf"(?:(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter"
    rf"(?:\s+of)?\s+{_YEAR}|q[1-4]\s+{_YEAR})"
)
_YEAR_PERIOD = rf"(?:in|during|by|through|throughout)\s+{_YEAR}"
_TIME_SIGNAL = re.compile(
    rf"\b(?:before|after|until|since|today|tomorrow|yesterday|as of|"
    rf"{_WEEKDAYS}|{_MONTHS}|{_QUARTER_PERIOD}|{_YEAR_PERIOD})\b|"
    r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b|"
    r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b|\b\d{1,2}\s*[ap]m\b|"
    r"截至|之前|之后|直到|今天|明天|昨日|周[一二三四五六日天]|"
    r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?",
    re.IGNORECASE,
)
_TEMPORAL_VALUE = re.compile(
    rf"\b(?:{_QUARTER_PERIOD}|{_YEAR_PERIOD})\b|"
    rf"\b(?:{_MONTHS})(?:\s+\d{{1,2}}(?:,?\s+\d{{4}})?|\s+{_YEAR})\b|"
    r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b|"
    r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b|\b\d{1,2}\s*[ap]m\b|"
    r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?",
    re.IGNORECASE,
)
_IRREGULAR_PREDICATES = (
    "arose|ate|became|began|broke|brought|built|bought|came|caught|chose|"
    "did|drew|drove|fell|felt|found|gave|got|grew|had|heard|held|kept|knew|"
    "led|left|lost|made|meant|met|paid|put|ran|read|said|saw|sent|set|sold|"
    "spoke|stood|taught|thought|told|took|understood|went|won|wrote"
)
_PREDICATE = re.compile(
    r"\b(?:is|are|was|were|be|been|has|have|had|do|does|did|will|would|"
    r"can|could|may|might|must|should|" + _IRREGULAR_PREDICATES +
    r"|[a-z]{3,}(?:ed|ing|s))\b",
    re.IGNORECASE,
)
_SCOPE_VERB_THAT = re.compile(
    r"\b(?:allege[ds]?|say|says|said|report(?:s|ed)?|claim(?:s|ed)?|"
    r"state[ds]?|announce[ds]?|believe[ds]?|expect(?:s|ed)?)\s+that\b",
    re.IGNORECASE,
)
_LEADING_SHARED_SCOPE = re.compile(
    r"^(?:\s*(?:in|at|on|near|within|outside|across|under|over|before|after|"
    r"according\s+to)\b|\s*(?:在|于|截至|据))",
    re.IGNORECASE,
)
_CORE_DIMENSIONS = {"actor_subject", "predicate_object", "scope_location"}
_POTENTIALLY_SHARED_DIMENSIONS = {
    "time", "negation", "modality", "comparison_baseline",
}
_CORE_ACTION_EQUIVALENTS = {
    "commence": "start", "commenced": "start", "commences": "start",
    "commencing": "start", "begin": "start", "began": "start",
    "begun": "start", "begins": "start", "beginning": "start",
    "start": "start", "started": "start", "starts": "start",
    "starting": "start",
}
_TEMPORAL_LINKER = re.compile(
    r"\b(?:in|during|by|through|throughout|on|at)\s+(?:the\s+)?$",
    re.IGNORECASE,
)


def _sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def _trim(text, start, end):
    while start < end and (text[start].isspace() or text[start] in ",，"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",，"):
        end -= 1
    return start, end


def _has_predicate(value):
    return bool(_PREDICATE.search(value) or re.search(
        r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}", value))


def _looks_like_clause(value):
    latin = re.findall(r"[A-Za-z]+", value)
    if len(latin) >= 2:
        return len(latin[-1]) >= 3 and latin[-1].islower()
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", value)) >= 4


def _has_independent_subject(value):
    """Require a visible subject before the first recognised Latin predicate."""
    if not re.search(r"[A-Za-z]", value):
        return True
    predicate = _PREDICATE.search(value)
    return bool(predicate and re.search(r"[A-Za-z]", value[:predicate.start()]))


def _compound_has_shared_scope(chunk, candidates, first_break_start):
    """Keep a compound intact when splitting could detach a shared modifier."""
    if _SCOPE_VERB_THAT.search(chunk[:first_break_start]):
        return True
    if _LEADING_SHARED_SCOPE.search(chunk):
        return True
    if "," in chunk[:first_break_start] or "，" in chunk[:first_break_start]:
        return True
    whole_dimensions = set(_dimensions(chunk))
    for dimension in _POTENTIALLY_SHARED_DIMENSIONS & whole_dimensions:
        if any(dimension not in _dimensions(value) for value in candidates):
            return True
    return False


def _dimensions(value):
    checks = ["actor_subject", "predicate_object", "scope_location"]
    without_temporal_values = _TEMPORAL_VALUE.sub("", value)
    if re.search(r"\d|%|％|\bpercent\b|百分之|[$€£¥￥]",
                 without_temporal_values, re.IGNORECASE):
        checks.append("quantity_unit_denominator")
    if _TIME_SIGNAL.search(value):
        checks.append("time")
    patterns = (
        ("negation", r"\b(?:not|no|never|without|neither)\b|未|没有|并非|不得|无"),
        ("condition", r"\b(?:if|unless|provided that|when)\b|如果|若|除非"),
        ("modality", r"\b(?:may|might|could|must|should|planned|expected|alleged)\b|"
                     r"可能|必须|应当|计划|预计|据称"),
        ("comparison_baseline", r"\b(?:more|less|than|versus|compared|increase|decrease)\b|"
                                r"相比|高于|低于|增加|减少"),
        ("attribution_causality", r"\b(?:according to|said|reported|because|caused by)\b|"
                                  r"据|表示|报道|因为|导致"),
    )
    for name, pattern in patterns:
        if re.search(pattern, value, re.IGNORECASE):
            checks.append(name)
    return checks


def build_target_plan(target):
    """Return stable, exact target spans that must each receive one result."""
    raw = asdict(target) if is_dataclass(target) else dict(target)
    text = raw.get("text")
    identifier = raw.get("id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("target plan needs nonempty target text")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("target plan needs nonempty target id")

    chunks = []
    cursor = 0
    for match in _SENTENCE_BREAK.finditer(text):
        start, end = _trim(text, cursor, match.start())
        if start < end:
            chunks.append((start, end))
        cursor = match.end()
    start, end = _trim(text, cursor, len(text))
    if start < end:
        chunks.append((start, end))
    if not chunks:
        start, end = _trim(text, 0, len(text))
        chunks = [(start, end)]
    # Period heuristics can see abbreviations as boundaries. Merge any fragment
    # without a predicate into its following clause (or the prior clause at
    # end) so "U.S." and "Dr." never become standalone obligations.
    normalized, pending_start = [], None
    for chunk_start, chunk_end in chunks:
        if pending_start is not None:
            chunk_start = pending_start
            pending_start = None
        candidate_text = text[chunk_start:chunk_end]
        abbreviation_prefix = bool(re.search(
            r"(?:^|[.\s])[A-Za-z]$", candidate_text))
        if abbreviation_prefix or not _has_predicate(candidate_text):
            pending_start = chunk_start
        else:
            normalized.append((chunk_start, chunk_end))
    if pending_start is not None:
        if normalized:
            normalized[-1] = (normalized[-1][0], chunks[-1][1])
        else:
            normalized = [(pending_start, chunks[-1][1])]
    chunks = normalized

    spans = []
    for chunk_start, chunk_end in chunks:
        chunk = text[chunk_start:chunk_end]
        if _CONDITIONAL.search(chunk):
            spans.append((chunk_start, chunk_end))
            continue
        candidates, local, breaks = [], 0, []
        for match in _CLAUSE_BREAK.finditer(chunk):
            breaks.append(match)
            left = _trim(text, chunk_start + local, chunk_start + match.start())
            if left[0] < left[1]:
                candidates.append(left)
            local = match.end()
        tail = _trim(text, chunk_start + local, chunk_end)
        if tail[0] < tail[1]:
            candidates.append(tail)
        candidate_texts = [text[item_start:item_end]
                           for item_start, item_end in candidates]
        shared_scope = bool(
            len(candidates) > 1 and breaks
            and _compound_has_shared_scope(
                chunk, candidate_texts, breaks[0].start()))
        if shared_scope:
            spans.append((chunk_start, chunk_end))
        elif len(candidates) > 1 and all(
                _has_predicate(value) and _has_independent_subject(value)
                for value in candidate_texts):
            spans.extend(candidates)
        elif len(candidates) > 1 and all(
                _looks_like_clause(value) for value in candidate_texts):
            raise ValueError(
                "ambiguous compound target cannot be safely split")
        else:
            spans.append((chunk_start, chunk_end))

    if len(spans) > MAX_PROBES:
        raise ValueError("target exceeds deterministic probe bound")
    probes = []
    for number, (probe_start, probe_end) in enumerate(spans, 1):
        probe_text = text[probe_start:probe_end]
        probe_id = identifier + ":probe:" + _sha(
            [identifier, probe_start, probe_end, probe_text])[:20]
        probes.append({
            "number": number,
            "id": probe_id,
            "text": probe_text,
            "start": probe_start,
            "end": probe_end,
            "required_dimensions": _dimensions(probe_text),
        })
    body = {
        "version": TARGET_PLAN_VERSION,
        "target_id": identifier,
        "target_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "operator": "all",
        "probes": probes,
    }
    body["plan_sha256"] = _sha(body)
    return body


def evidence_terms(value):
    """High-signal overlap gate for a probe and its purported exact basis."""
    stop = {"the", "and", "that", "this", "with", "from", "before", "after",
            "until", "has", "have", "had", "was", "were", "will"}
    terms = set(re.findall(r"[a-z0-9%]{3,}", value.lower())) - stop
    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", value):
        widths = (1,) if len(sequence) == 1 else (2, 3)
        for width in widths:
            terms.update(sequence[index:index + width]
                         for index in range(len(sequence) - width + 1))
    return terms


def _word_forms(token):
    token = token.lower()
    forms = {token}
    if token.endswith("ing") and len(token) > 5:
        root = token[:-3]
        forms.update((root, root + "e"))
    if token.endswith("ed") and len(token) > 4:
        root = token[:-2]
        forms.update((root, root + "e"))
    if token.endswith("es") and len(token) > 4:
        root = token[:-2]
        forms.update((root, root + "e", token[:-1]))
    elif token.endswith("s") and len(token) > 3:
        forms.add(token[:-1])
    return forms


def _content_tokens(value):
    ignored = {
        "the", "and", "that", "this", "with", "from", "before", "after",
        "until", "since", "today", "tomorrow", "yesterday", "well", "also",
        "for", "into", "onto", "over", "under", "between", "during", "about",
        "will", "would", "can", "could", "may", "might", "must", "should",
        "has", "have", "had", "was", "were", "are", "does", "did", "not",
    }
    return [token for token in re.findall(r"[a-z]{3,}", value.lower())
            if token not in ignored]


def _claim_tokens(value):
    """Surface tokens whose order preserves actors and scope binding."""
    ignored = {"the", "a", "an"}
    return [token for token in re.findall(r"[a-z]+", value.lower())
            if token not in ignored]


def _without_temporal_surface(value):
    """Blank only target-plan-recognised temporal spans and their linkers."""
    covered = [False] * len(value)
    for pattern in (_TEMPORAL_VALUE, _TIME_SIGNAL):
        for match in pattern.finditer(value):
            start, end = match.span()
            linker = _TEMPORAL_LINKER.search(value[:start])
            if linker is not None:
                start = linker.start()
            covered[start:end] = [True] * (end - start)
    return "".join(" " if hidden else character
                   for character, hidden in zip(value, covered))


def _core_claim_tokens(value):
    """Canonical core surface while retaining role and conjunction order."""
    tokens = _claim_tokens(_without_temporal_surface(value))
    result = []
    for index, token in enumerate(tokens):
        # A retrospective outcome can support the future action without
        # repeating its auxiliary, but do not discard auxiliaries generally.
        if (token in {"will", "would"} and index + 1 < len(tokens)
                and tokens[index + 1] in _CORE_ACTION_EQUIVALENTS):
            continue
        result.append(_CORE_ACTION_EQUIVALENTS.get(token, token))
    return result


def _core_content_tokens(value):
    return [_CORE_ACTION_EQUIVALENTS.get(token, token)
            for token in _content_tokens(_without_temporal_surface(value))]


def _target_identity_tokens(value):
    """Conservative ordered actor/object anchors with the predicate removed."""
    surface = _without_temporal_surface(value)
    auxiliaries = {
        "is", "are", "was", "were", "be", "been", "has", "have", "had",
        "do", "does", "did", "will", "would", "can", "could", "may",
        "might", "must", "should",
    }
    candidates = list(_PREDICATE.finditer(surface))
    # Suffix heuristics can mistake a capitalized subject such as "Boeing"
    # or "Officials" for a verb. Prefer an explicit auxiliary; otherwise
    # require a visible subject before the predicate when one exists.
    predicate = next(
        (item for item in candidates if item.group().lower() in auxiliaries),
        None)
    if predicate is None:
        predicate = next(
            (item for item in candidates
             if _content_tokens(surface[:item.start()])),
            candidates[0] if candidates else None)
    if predicate is None:
        return _core_content_tokens(surface)
    before = _core_content_tokens(surface[:predicate.start()])
    after = _core_content_tokens(surface[predicate.end():])
    if predicate.group().lower() in auxiliaries and after:
        # The first content word after an auxiliary is the predicate head.
        after = after[1:]
    return before + after


def _ordered_identity_surface(probe_text, basis_text):
    anchors = _target_identity_tokens(probe_text)
    return not anchors or _ordered_token_match(
        anchors, _core_content_tokens(basis_text))


def _ordered_token_match(target_tokens, basis_tokens):
    cursor = 0
    for target_token in target_tokens:
        target_forms = _word_forms(target_token)
        while cursor < len(basis_tokens):
            current = basis_tokens[cursor]
            cursor += 1
            if target_forms & _word_forms(current):
                break
        else:
            return False
    return True


def _ordered_supported_surface(probe_text, basis_text):
    """Fail closed on role reversal and detached conjunction scope."""
    target_tokens = _core_claim_tokens(probe_text)
    if target_tokens and not _ordered_token_match(
            target_tokens, _core_claim_tokens(basis_text)):
        return False
    target_cjk = re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff]", _without_temporal_surface(probe_text))
    basis_cjk = re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff]", _without_temporal_surface(basis_text))
    if target_cjk and not _ordered_token_match(target_cjk, basis_cjk):
        return False
    return True


def _predicate_markers(value):
    match = _PREDICATE.search(value)
    if match is None:
        return set()
    verb = match.group().lower()
    auxiliaries = {
        "is", "are", "was", "were", "be", "been", "has", "have", "had",
        "do", "does", "did", "will", "would", "can", "could", "may",
        "might", "must", "should",
    }
    if verb not in auxiliaries:
        return _word_forms(verb)
    tail = re.findall(r"[a-z]{3,}", value[match.end():].lower())
    ignored = {"not", "never", "before", "after", "until", "since",
               "today", "tomorrow", "yesterday", "than", "with", "from"}
    markers = set().union(*(_word_forms(token) for token in tail
                            if token not in ignored))
    return markers or {verb}


def _temporal_markers(value):
    lowered = value.lower()
    markers = set(re.findall(
        rf"\b(?:before|after|until|since|today|tomorrow|yesterday|"
        rf"{_WEEKDAYS}|{_MONTHS})\b", lowered))
    for match in _TEMPORAL_VALUE.finditer(value):
        matched = match.group().lower()
        # Quarter/year phrases are normalized below so equivalent calendar
        # expressions such as "second quarter" and "June" can agree.
        if re.fullmatch(
                rf"(?:{_QUARTER_PERIOD}|{_YEAR_PERIOD})", matched,
                re.IGNORECASE):
            continue
        markers.add(re.sub(r"\W+", "", matched))
    markers.update(re.findall(
        r"截至|之前|之后|直到|今天|明天|昨日|周[一二三四五六日天]|"
        r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?", value))
    if _TIME_SIGNAL.search(value):
        markers.update("year:" + year for year in re.findall(
            rf"\b({_YEAR})\b", lowered))
    quarter_names = {
        "first": "1", "1st": "1", "second": "2", "2nd": "2",
        "third": "3", "3rd": "3", "fourth": "4", "4th": "4",
    }
    for match in re.finditer(
            r"\b(?:(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter|"
            r"q([1-4]))\b", lowered):
        number = match.group(2) or quarter_names[match.group(1)]
        markers.add("quarter:" + number)
    month_quarters = {
        "january": "1", "february": "1", "march": "1",
        "april": "2", "may": "2", "june": "2",
        "july": "3", "august": "3", "september": "3",
        "october": "4", "november": "4", "december": "4",
    }
    for month in re.findall(rf"\b({_MONTHS})\b", lowered):
        markers.add("quarter:" + month_quarters[month])
    return {marker for marker in markers if marker}


def _quantity_markers(value):
    lowered = value.lower()
    markers = {"number:" + number
               for number in re.findall(r"\d+(?:\.\d+)?", lowered)}
    currency = {
        "$": "currency:usd", "usd": "currency:usd", "dollar": "currency:usd",
        "€": "currency:eur", "eur": "currency:eur", "euro": "currency:eur",
        "£": "currency:gbp", "gbp": "currency:gbp", "pound": "currency:gbp",
        "¥": "currency:yen-yuan", "￥": "currency:yen-yuan",
        "yuan": "currency:yen-yuan", "yen": "currency:yen-yuan",
    }
    for token, marker in currency.items():
        if token in lowered:
            markers.add(marker)
    if re.search(r"%|％|\bpercent(?:age)?\b|百分之", lowered):
        markers.add("unit:percent")
    for unit in re.findall(
            r"\b(?:thousand|million|billion|trillion|kg|kilogram|kilograms|"
            r"km|kilometer|kilometers|mile|miles|hour|hours|day|days|"
            r"people|persons|users|shares|votes)\b", lowered):
        markers.add("unit:" + unit.rstrip("s"))
    for denominator in re.findall(
            r"\bper\s+([a-z]+)|/\s*([a-z]+)|\beach\s+([a-z]+)", lowered):
        value = next(item for item in denominator if item)
        markers.add("denominator:" + value.rstrip("s"))
    return markers


def dimension_evidence_is_grounded(dimension, probe_text, basis_text,
                                   verdict="supported"):
    """Conservative structural gate for qualifier-specific basis."""
    probe_terms = evidence_terms(probe_text)
    basis_terms = evidence_terms(basis_text)
    quantity_terms = {term for term in probe_terms
                      if any(character.isdigit() for character in term)
                      or term in {"percent", "percentage"}}
    qualifier_terms = {
        "time": {"before", "after", "until", "since", "today", "tomorrow",
                 "yesterday", "monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday", "january", "february",
                 "march", "april", "june", "july", "august", "september",
                 "october", "november", "december"},
        "quantity_unit_denominator": quantity_terms,
        "modality": {"may", "might", "could", "must", "should", "planned",
                     "expected", "alleged"},
        "comparison_baseline": {"more", "less", "than", "versus", "compared",
                                "increase", "decrease", "higher", "lower"},
        "attribution_causality": {"according", "said", "reported", "because",
                                  "caused", "source", "statement"},
    }.get(dimension, set())
    anchor_patterns = {
        "time": r"\b(?:before|after|until|since|today|tomorrow|yesterday|"
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
                r"截至|之前|之后|直到|今天|明天|昨日|周[一二三四五六日天]",
        "negation": r"\b(?:not|no|never|without|neither)\b|未|没有|并非|不得|无",
        "condition": r"\b(?:if|unless|provided that|when)\b|如果|若|除非",
    }
    if dimension in anchor_patterns:
        for match in re.finditer(anchor_patterns[dimension], probe_text,
                                 re.IGNORECASE):
            qualifier_terms.update(evidence_terms(match.group()))
    core_terms = probe_terms - qualifier_terms
    if not (core_terms or probe_terms) & basis_terms:
        return False
    if verdict == "supported" and dimension in {
            "actor_subject", "predicate_object", "scope_location"}:
        if not _ordered_supported_surface(probe_text, basis_text):
            return False
        basis_forms = set().union(
            *(_word_forms(token) for token in _core_content_tokens(basis_text)))
        if any(not (_word_forms(token) & basis_forms)
               for token in _core_content_tokens(probe_text)):
            return False
    if verdict in {"contradicted", "conflicting"}:
        # A contrary qualifier or action must still bind to the target's
        # ordered actor/object identity. An unrelated filing date is not a
        # refutation merely because the issuer name overlaps.
        if not _ordered_identity_surface(probe_text, basis_text):
            return False
    if dimension == "predicate_object" and verdict == "supported":
        markers = _predicate_markers(probe_text)
        basis_stems = set().union(*(
            _word_forms(token) for token in re.findall(
                r"[a-z]{3,}", basis_text.lower())))
        if markers and not markers & basis_stems:
            return False
    if dimension == "time" and verdict == "supported":
        target_markers = _temporal_markers(probe_text)
        basis_markers = _temporal_markers(basis_text)
        if target_markers and not target_markers <= basis_markers:
            return False
    if dimension == "quantity_unit_denominator" and verdict == "supported":
        target_quantity = _quantity_markers(probe_text)
        basis_quantity = _quantity_markers(basis_text)
        if target_quantity and not target_quantity <= basis_quantity:
            return False
    signals = {
        "quantity_unit_denominator": r"\d|%|％|\bpercent\b|百分之|[$€£¥￥]",
        "time": r"\b(?:before|after|until|since|today|tomorrow|yesterday|"
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                r"january|february|march|april|may|june|july|august|september|"
                r"october|november|december)\b|\d{1,4}[-/:]\d|"
                r"截至|之前|之后|直到|今天|明天|昨日|周[一二三四五六日天]",
        "modality": r"\b(?:may|might|could|must|should|planned|expected|alleged|"
                    r"implemented|occurred|confirmed)\b|可能|必须|应当|计划|预计|据称|实施|发生|确认",
        "comparison_baseline": r"\b(?:more|less|than|versus|compared|increase|decrease|"
                               r"higher|lower)\b|相比|高于|低于|增加|减少|\d",
        "attribution_causality": r"\b(?:according to|said|reported|because|caused by|"
                                 r"source|statement)\b|据|表示|报道|因为|导致|来源|声明",
    }
    pattern = signals.get(dimension)
    if pattern is not None and not re.search(pattern, basis_text, re.IGNORECASE):
        return False
    if dimension == "negation" and verdict == "supported":
        return bool(re.search(
            r"\b(?:not|no|never|without|neither)\b|未|没有|并非|不得|无",
            basis_text, re.IGNORECASE))
    if dimension == "condition":
        return bool(re.search(
            r"\b(?:if|unless|provided that|when)\b|如果|若|除非",
            basis_text, re.IGNORECASE))
    return True


__all__ = ["TARGET_PLAN_VERSION", "build_target_plan", "evidence_terms",
           "dimension_evidence_is_grounded"]
