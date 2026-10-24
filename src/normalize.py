"""Text normalisation for business names and addresses (country-agnostic rules
plus abbreviation tables for US / India / France). No external lookups."""
import re
from functools import lru_cache
from anyascii import anyascii

TOKEN_RE = re.compile(r"[a-z0-9]+")

# ---------------------------------------------------------------- names
DBA_RE = re.compile(
    r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|trading as|t\s*/\s*a|also known as|aka|operating as)\b[:\s]*",
    re.I)

LEGAL = {
    # english / india
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "plc", "pllc", "pc", "pa", "the", "and", "of",
    # france
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "scp", "scm", "selarl", "cie", "et", "de",
    "du", "des", "la", "le", "les", "l", "d",
    # web noise and Hindi back-transliterations of legal forms
    "www", "com", "net", "org", "praivet", "piraivet", "praiveta", "privet", "limitet", "limitd",
    "limted", "elelpi", "dba", "pra", "li", "prai",
}
NAME_CANON = {
    "incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd",
    "private": "pvt", "pvt": "pvt", "compagnie": "cie", "&": "and", "n": "and",
    "intl": "international", "mgmt": "management", "svcs": "services", "svc": "services",
    "assoc": "association", "assn": "association", "ctr": "center", "centre": "center",
    "grp": "group", "hldgs": "holdings", "tech": "technology", "technologies": "technology",
    "bros": "brothers", "mfg": "manufacturing", "dept": "department", "univ": "university",
    "natl": "national", "st": "saint", "ste": "sainte",
}


def ascii_lower(s):
    if s is None:
        return ""
    return anyascii(s).lower()


@lru_cache(maxsize=2_000_000)
def norm_name(s):
    """-> (full normalised name, core name without legal/stop tokens, dba-stripped core)."""
    s = ascii_lower(s).replace("&", " and ")
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", lambda m: " " + m.group(0)[1:-1] + " ", s)
    # collapse dotted acronyms: s.a.s -> sas, p. ltd
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)
    parts = DBA_RE.split(s)
    toks = [NAME_CANON.get(t, t) for t in TOKEN_RE.findall(s)]
    full = " ".join(toks)
    core = " ".join(t for t in toks if t not in LEGAL) or full
    if len(parts) > 1:
        dt = [NAME_CANON.get(t, t) for t in TOKEN_RE.findall(parts[-1])]
        dba = " ".join(t for t in dt if t not in LEGAL) or core
    else:
        dba = core
    return full, core, dba


# ---------------------------------------------------------------- addresses
ADDR_CANON = {
    # US / generic english
    "street": "st", "saint": "st", "str": "st", "avenue": "ave", "av": "ave", "aven": "ave",
    "anenue": "ave", "road": "rd", "drive": "dr", "boulevard": "blvd", "blv": "blvd",
    "lane": "ln", "court": "ct", "place": "pl", "circle": "cir", "highway": "hwy",
    "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq", "suite": "ste",
    "apartment": "apt", "unit": "unit", "building": "bldg", "floor": "fl", "north": "n",
    "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "mount": "mt", "fort": "ft", "expressway": "expy",
    "turnpike": "tpke", "crossing": "xing", "point": "pt", "heights": "hts", "center": "ctr",
    "centre": "ctr", "way": "way", "pike": "pike", "number": "no", "num": "no",
    # india
    "nagar": "ngr", "marg": "rd", "salai": "rd", "sadak": "rd", "chowk": "chk", "colony": "col",
    "sector": "sec", "phase": "ph", "near": "nr", "opposite": "opp", "opp": "opp", "behind": "bhd",
    "house": "h", "plot": "plot", "cross": "crs", "main": "main", "block": "blk", "khasra": "kh",
    "village": "vill", "vpo": "vill", "post": "po", "district": "dist", "taluk": "tq",
    "tehsil": "tq", "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "gurgaon": "gurugram",
    # france
    "rue": "r", "avenue": "ave", "boulevard": "blvd", "bd": "blvd", "place": "pl", "allee": "all",
    "chemin": "che", "chem": "che", "impasse": "imp", "route": "rte", "quai": "qu", "cours": "crs",
    "faubourg": "fbg", "residence": "res", "lieu": "ld", "lieudit": "ld", "bis": "bis", "ter": "ter",
    "cedex": "", "cite": "cite", "square": "sq", "passage": "pas", "promenade": "prom",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
US_MULTI = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "district of columbia": "dc",
}
FR_ADMIN = {"hauts de france", "nouvelle aquitaine", "pays de la loire", "nord", "gironde", "loire atlantique",
            "pas de calais", "ile de france", "occitanie", "bretagne", "normandie", "grand est", "corse",
            "auvergne rhone alpes", "provence alpes cote d'azur", "bourgogne franche comte", "centre val de loire"}
NULLISH = {"null", "none", "na", "n/a", "nan", "unknown", "-"}
GENERIC_ADDR = set(ADDR_CANON.values()) | {"no", "nr", "unit", "apt", "ste", "fl", "floor", "po",
                                           "box", "pmb", "bldg", "cdp", "city", "of", "the", "de",
                                           "du", "des", "la", "le", "les", "d", "l", "and"}


@lru_cache(maxsize=2_000_000)
def norm_addr(s, country=""):
    """-> dict with normalised address tokens, numbers, and word tokens."""
    s = ascii_lower(s)
    if country == "US":
        for k, v in US_MULTI.items():
            if k in s:
                s = s.replace(k, v)
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)  # 19th -> 19
    raw_parts = [p.strip() for p in s.split(",")]
    if country == "France":  # region / department names vary between sources (Hauts-de-France vs Nord): drop them,
        # like US state names are normalised; only whole comma-separated parts are removed, never city names
        raw_parts = [p for p in raw_parts if " ".join(p.replace("-", " ").split()) not in FR_ADMIN]
        s = ", ".join(raw_parts)
    toks = []
    for t in TOKEN_RE.findall(s):
        if t in NULLISH:
            continue
        if country == "US" and t in US_STATES:
            t = US_STATES[t]
        else:
            t = ADDR_CANON.get(t, t)
        if t:
            toks.append(t)
    nums = []
    for t in toks:
        if t.isdigit():
            nums.append(t.lstrip("0") or "0")
        else:  # mixed like b31, 8k, 570/13 pieces
            d = re.findall(r"\d+", t)
            nums.extend(x.lstrip("0") or "0" for x in d)
    words = [t for t in toks if not any(c.isdigit() for c in t)]
    return " ".join(toks), tuple(nums), tuple(words), tuple(p for p in raw_parts if p and p not in NULLISH)


def skeleton(tok):
    """consonant skeleton: helps transliteration variants (ram/raam, marketimg/marketing)."""
    if not tok:
        return tok
    s = tok[0] + re.sub(r"[aeiouyhw]", "", tok[1:])
    return re.sub(r"(.)\1+", r"\1", s)
