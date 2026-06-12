"""
Real FIFA World Cup 2026 knockout-stage bracket structure (Matches 73-104).

Source: FIFA Tournament Regulations (Round of 32 matchup definitions and the
Annex C third-place qualification table) and the official bracket tree, as
published on Wikipedia's "2026 FIFA World Cup knockout stage" article.
Re-verified June 2026 directly against the live article: R32_MATCHES,
BRACKET_TREE, LEFT_R32_ORDER/RIGHT_R32_ORDER, FINAL_MATCH (104) and
THIRD_PLACE_MATCH (103) all match the published bracket exactly. Match
dates/venues are best-effort, cross-referenced across multiple public
schedules.

This module is pure data + pure functions -- no randomness, no Streamlit,
no model imports -- so it can be shared by both the Monte Carlo simulator
(models/monte_carlo.py) and the dashboard's bracket renderer
(dashboard/app.py).

-------------------------------------------------------------------------
"Best third-placed team" slots and Annex C
-------------------------------------------------------------------------
8 of the 16 Round-of-32 matches pair a group winner against the "best
third-placed team" from a published set of 5 groups. Which actual group's
3rd-placed team lands in which of these 8 slots depends on which 8 (of 12)
groups produce a qualifying 3rd-placed team -- FIFA pre-computed this for
all C(12, 8) = 495 combinations in "Annex C" of the tournament regulations,
subject to one rule: a 3rd-placed team can never be drawn against a team
from its own group (which is why the published "best 3rd place" eligible
sets below never include the anchor group itself).

`resolve_third_place_assignment` is a literal lookup into ANNEX_C_TABLE,
FIFA's full 495-row published Annex C table (scraped from Wikipedia's "2026
FIFA World Cup knockout stage" article and validated: every row is a
permutation of its 8 qualifying groups, every assignment respects
THIRD_PLACE_ELIGIBLE, and all 495 = C(12,8) combinations of groups A-L are
present with zero duplicates and zero missing combinations). This replaces
the previous backtracking implementation, which only reproduced FIFA's
exact published choice in 32/495 (6.5%) of cases -- it satisfied the two
hard constraints (each group used once, never against its own anchor) but
not FIFA's specific tie-break choice among the multiple assignments that
also satisfy those constraints.
"""

from typing import Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Round of 32 (Matches 73-88)
#
# Slot codes:
#   "1X"            -> winner of Group X
#   "2X"            -> runner-up of Group X
#   "3:A,B,C,D,E"   -> best qualifying 3rd-placed team from groups A/B/C/D/E
# ---------------------------------------------------------------------------
R32_MATCHES: Dict[int, Tuple[str, str]] = {
    73: ("2A", "2B"),
    74: ("1E", "3:A,B,C,D,F"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3:C,D,F,G,H"),
    78: ("2E", "2I"),
    79: ("1A", "3:C,E,F,H,I"),
    80: ("1L", "3:E,H,I,J,K"),
    81: ("1D", "3:B,E,F,I,J"),
    82: ("1G", "3:A,E,H,I,J"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3:E,F,G,I,J"),
    86: ("1J", "2H"),
    87: ("1K", "3:D,E,I,J,L"),
    88: ("2D", "2G"),
}

# Eligible "best 3rd place" group sets, keyed by the R32 match that has a
# "3:..." slot. Derived directly from R32_MATCHES.
THIRD_PLACE_ELIGIBLE: Dict[int, frozenset] = {
    m: frozenset(slots[1][2:].split(","))
    for m, slots in R32_MATCHES.items()
    if slots[1].startswith("3:")
}
THIRD_PLACE_SLOTS: List[int] = list(THIRD_PLACE_ELIGIBLE.keys())  # [74,77,79,80,81,82,85,87]

# ---------------------------------------------------------------------------
# Annex C: FIFA's published table of third-place qualification assignments
# ---------------------------------------------------------------------------
# Column order, matching Annex C's published header
# ("1A vs; 1B vs; 1D vs; 1E vs; 1G vs; 1I vs; 1K vs; 1L vs"):
ANNEX_C_COLUMN_TO_MATCH: Tuple[int, ...] = (79, 85, 81, 74, 82, 77, 87, 80)

# Each line: "<8 sorted qualifying-group letters>:<8-letter assignment in
# ANNEX_C_COLUMN_TO_MATCH column order>". All 495 = C(12,8) combinations of
# groups A-L, taken verbatim from FIFA's published Annex C table (scraped
# from Wikipedia's "2026 FIFA World Cup knockout stage" article, June 2026).
_ANNEX_C_RAW = """\
EFGHIJKL:EJIFHGLK
DFGHIJKL:HGIDJFLK
DEGHIJKL:EJIDHGLK
DEFHIJKL:EJIDHFLK
DEFGIJKL:EGIDJFLK
DEFGHJKL:EGJDHFLK
DEFGHIKL:EGIDHFLK
DEFGHIJL:EGJDHFLI
DEFGHIJK:EGJDHFIK
CFGHIJKL:HGICJFLK
CEGHIJKL:EJICHGLK
CEFHIJKL:EJICHFLK
CEFGIJKL:EGICJFLK
CEFGHJKL:EGJCHFLK
CEFGHIKL:EGICHFLK
CEFGHIJL:EGJCHFLI
CEFGHIJK:EGJCHFIK
CDGHIJKL:HGICJDLK
CDFHIJKL:CJIDHFLK
CDFGIJKL:CGIDJFLK
CDFGHJKL:CGJDHFLK
CDFGHIKL:CGIDHFLK
CDFGHIJL:CGJDHFLI
CDFGHIJK:CGJDHFIK
CDEHIJKL:EJICHDLK
CDEGIJKL:EGICJDLK
CDEGHJKL:EGJCHDLK
CDEGHIKL:EGICHDLK
CDEGHIJL:EGJCHDLI
CDEGHIJK:EGJCHDIK
CDEFIJKL:CJEDIFLK
CDEFHJKL:CJEDHFLK
CDEFHIKL:CEIDHFLK
CDEFHIJL:CJEDHFLI
CDEFHIJK:CJEDHFIK
CDEFGJKL:CGEDJFLK
CDEFGIKL:CGEDIFLK
CDEFGIJL:CGEDJFLI
CDEFGIJK:CGEDJFIK
CDEFGHKL:CGEDHFLK
CDEFGHJL:CGJDHFLE
CDEFGHJK:CGJDHFEK
CDEFGHIL:CGEDHFLI
CDEFGHIK:CGEDHFIK
CDEFGHIJ:CGJDHFEI
BFGHIJKL:HJBFIGLK
BEGHIJKL:EJIBHGLK
BEFHIJKL:EJBFIHLK
BEFGIJKL:EJBFIGLK
BEFGHJKL:EJBFHGLK
BEFGHIKL:EGBFIHLK
BEFGHIJL:EJBFHGLI
BEFGHIJK:EJBFHGIK
BDGHIJKL:HJBDIGLK
BDFHIJKL:HJBDIFLK
BDFGIJKL:IGBDJFLK
BDFGHJKL:HGBDJFLK
BDFGHIKL:HGBDIFLK
BDFGHIJL:HGBDJFLI
BDFGHIJK:HGBDJFIK
BDEHIJKL:EJBDIHLK
BDEGIJKL:EJBDIGLK
BDEGHJKL:EJBDHGLK
BDEGHIKL:EGBDIHLK
BDEGHIJL:EJBDHGLI
BDEGHIJK:EJBDHGIK
BDEFIJKL:EJBDIFLK
BDEFHJKL:EJBDHFLK
BDEFHIKL:EIBDHFLK
BDEFHIJL:EJBDHFLI
BDEFHIJK:EJBDHFIK
BDEFGJKL:EGBDJFLK
BDEFGIKL:EGBDIFLK
BDEFGIJL:EGBDJFLI
BDEFGIJK:EGBDJFIK
BDEFGHKL:EGBDHFLK
BDEFGHJL:HGBDJFLE
BDEFGHJK:HGBDJFEK
BDEFGHIL:EGBDHFLI
BDEFGHIK:EGBDHFIK
BDEFGHIJ:HGBDJFEI
BCGHIJKL:HJBCIGLK
BCFHIJKL:HJBCIFLK
BCFGIJKL:IGBCJFLK
BCFGHJKL:HGBCJFLK
BCFGHIKL:HGBCIFLK
BCFGHIJL:HGBCJFLI
BCFGHIJK:HGBCJFIK
BCEHIJKL:EJBCIHLK
BCEGIJKL:EJBCIGLK
BCEGHJKL:EJBCHGLK
BCEGHIKL:EGBCIHLK
BCEGHIJL:EJBCHGLI
BCEGHIJK:EJBCHGIK
BCEFIJKL:EJBCIFLK
BCEFHJKL:EJBCHFLK
BCEFHIKL:EIBCHFLK
BCEFHIJL:EJBCHFLI
BCEFHIJK:EJBCHFIK
BCEFGJKL:EGBCJFLK
BCEFGIKL:EGBCIFLK
BCEFGIJL:EGBCJFLI
BCEFGIJK:EGBCJFIK
BCEFGHKL:EGBCHFLK
BCEFGHJL:HGBCJFLE
BCEFGHJK:HGBCJFEK
BCEFGHIL:EGBCHFLI
BCEFGHIK:EGBCHFIK
BCEFGHIJ:HGBCJFEI
BCDHIJKL:HJBCIDLK
BCDGIJKL:IGBCJDLK
BCDGHJKL:HGBCJDLK
BCDGHIKL:HGBCIDLK
BCDGHIJL:HGBCJDLI
BCDGHIJK:HGBCJDIK
BCDFIJKL:CJBDIFLK
BCDFHJKL:CJBDHFLK
BCDFHIKL:CIBDHFLK
BCDFHIJL:CJBDHFLI
BCDFHIJK:CJBDHFIK
BCDFGJKL:CGBDJFLK
BCDFGIKL:CGBDIFLK
BCDFGIJL:CGBDJFLI
BCDFGIJK:CGBDJFIK
BCDFGHKL:CGBDHFLK
BCDFGHJL:CGBDHFLJ
BCDFGHJK:HGBCJFDK
BCDFGHIL:CGBDHFLI
BCDFGHIK:CGBDHFIK
BCDFGHIJ:HGBCJFDI
BCDEIJKL:EJBCIDLK
BCDEHJKL:EJBCHDLK
BCDEHIKL:EIBCHDLK
BCDEHIJL:EJBCHDLI
BCDEHIJK:EJBCHDIK
BCDEGJKL:EGBCJDLK
BCDEGIKL:EGBCIDLK
BCDEGIJL:EGBCJDLI
BCDEGIJK:EGBCJDIK
BCDEGHKL:EGBCHDLK
BCDEGHJL:HGBCJDLE
BCDEGHJK:HGBCJDEK
BCDEGHIL:EGBCHDLI
BCDEGHIK:EGBCHDIK
BCDEGHIJ:HGBCJDEI
BCDEFJKL:CJBDEFLK
BCDEFIKL:CEBDIFLK
BCDEFIJL:CJBDEFLI
BCDEFIJK:CJBDEFIK
BCDEFHKL:CEBDHFLK
BCDEFHJL:CJBDHFLE
BCDEFHJK:CJBDHFEK
BCDEFHIL:CEBDHFLI
BCDEFHIK:CEBDHFIK
BCDEFHIJ:CJBDHFEI
BCDEFGKL:CGBDEFLK
BCDEFGJL:CGBDJFLE
BCDEFGJK:CGBDJFEK
BCDEFGIL:CGBDEFLI
BCDEFGIK:CGBDEFIK
BCDEFGIJ:CGBDJFEI
BCDEFGHL:CGBDHFLE
BCDEFGHK:CGBDHFEK
BCDEFGHJ:HGBCJFDE
BCDEFGHI:CGBDHFEI
AFGHIJKL:HJIFAGLK
AEGHIJKL:EJIAHGLK
AEFHIJKL:EJIFAHLK
AEFGIJKL:EJIFAGLK
AEFGHJKL:EGJFAHLK
AEFGHIKL:EGIFAHLK
AEFGHIJL:EGJFAHLI
AEFGHIJK:EGJFAHIK
ADGHIJKL:HJIDAGLK
ADFHIJKL:HJIDAFLK
ADFGIJKL:IGJDAFLK
ADFGHJKL:HGJDAFLK
ADFGHIKL:HGIDAFLK
ADFGHIJL:HGJDAFLI
ADFGHIJK:HGJDAFIK
ADEHIJKL:EJIDAHLK
ADEGIJKL:EJIDAGLK
ADEGHJKL:EGJDAHLK
ADEGHIKL:EGIDAHLK
ADEGHIJL:EGJDAHLI
ADEGHIJK:EGJDAHIK
ADEFIJKL:EJIDAFLK
ADEFHJKL:HJEDAFLK
ADEFHIKL:HEIDAFLK
ADEFHIJL:HJEDAFLI
ADEFHIJK:HJEDAFIK
ADEFGJKL:EGJDAFLK
ADEFGIKL:EGIDAFLK
ADEFGIJL:EGJDAFLI
ADEFGIJK:EGJDAFIK
ADEFGHKL:HGEDAFLK
ADEFGHJL:HGJDAFLE
ADEFGHJK:HGJDAFEK
ADEFGHIL:HGEDAFLI
ADEFGHIK:HGEDAFIK
ADEFGHIJ:HGJDAFEI
ACGHIJKL:HJICAGLK
ACFHIJKL:HJICAFLK
ACFGIJKL:IGJCAFLK
ACFGHJKL:HGJCAFLK
ACFGHIKL:HGICAFLK
ACFGHIJL:HGJCAFLI
ACFGHIJK:HGJCAFIK
ACEHIJKL:EJICAHLK
ACEGIJKL:EJICAGLK
ACEGHJKL:EGJCAHLK
ACEGHIKL:EGICAHLK
ACEGHIJL:EGJCAHLI
ACEGHIJK:EGJCAHIK
ACEFIJKL:EJICAFLK
ACEFHJKL:HJECAFLK
ACEFHIKL:HEICAFLK
ACEFHIJL:HJECAFLI
ACEFHIJK:HJECAFIK
ACEFGJKL:EGJCAFLK
ACEFGIKL:EGICAFLK
ACEFGIJL:EGJCAFLI
ACEFGIJK:EGJCAFIK
ACEFGHKL:HGECAFLK
ACEFGHJL:HGJCAFLE
ACEFGHJK:HGJCAFEK
ACEFGHIL:HGECAFLI
ACEFGHIK:HGECAFIK
ACEFGHIJ:HGJCAFEI
ACDHIJKL:HJICADLK
ACDGIJKL:IGJCADLK
ACDGHJKL:HGJCADLK
ACDGHIKL:HGICADLK
ACDGHIJL:HGJCADLI
ACDGHIJK:HGJCADIK
ACDFIJKL:CJIDAFLK
ACDFHJKL:HJFCADLK
ACDFHIKL:HFICADLK
ACDFHIJL:HJFCADLI
ACDFHIJK:HJFCADIK
ACDFGJKL:CGJDAFLK
ACDFGIKL:CGIDAFLK
ACDFGIJL:CGJDAFLI
ACDFGIJK:CGJDAFIK
ACDFGHKL:HGFCADLK
ACDFGHJL:CGJDAFLH
ACDFGHJK:HGJCAFDK
ACDFGHIL:HGFCADLI
ACDFGHIK:HGFCADIK
ACDFGHIJ:HGJCAFDI
ACDEIJKL:EJICADLK
ACDEHJKL:HJECADLK
ACDEHIKL:HEICADLK
ACDEHIJL:HJECADLI
ACDEHIJK:HJECADIK
ACDEGJKL:EGJCADLK
ACDEGIKL:EGICADLK
ACDEGIJL:EGJCADLI
ACDEGIJK:EGJCADIK
ACDEGHKL:HGECADLK
ACDEGHJL:HGJCADLE
ACDEGHJK:HGJCADEK
ACDEGHIL:HGECADLI
ACDEGHIK:HGECADIK
ACDEGHIJ:HGJCADEI
ACDEFJKL:CJEDAFLK
ACDEFIKL:CEIDAFLK
ACDEFIJL:CJEDAFLI
ACDEFIJK:CJEDAFIK
ACDEFHKL:HEFCADLK
ACDEFHJL:HJFCADLE
ACDEFHJK:HJECAFDK
ACDEFHIL:HEFCADLI
ACDEFHIK:HEFCADIK
ACDEFHIJ:HJECAFDI
ACDEFGKL:CGEDAFLK
ACDEFGJL:CGJDAFLE
ACDEFGJK:CGJDAFEK
ACDEFGIL:CGEDAFLI
ACDEFGIK:CGEDAFIK
ACDEFGIJ:CGJDAFEI
ACDEFGHL:HGFCADLE
ACDEFGHK:HGECAFDK
ACDEFGHJ:HGJCAFDE
ACDEFGHI:HGECAFDI
ABGHIJKL:HJBAIGLK
ABFHIJKL:HJBAIFLK
ABFGIJKL:IJBFAGLK
ABFGHJKL:HJBFAGLK
ABFGHIKL:HGBAIFLK
ABFGHIJL:HJBFAGLI
ABFGHIJK:HJBFAGIK
ABEHIJKL:EJBAIHLK
ABEGIJKL:EJBAIGLK
ABEGHJKL:EJBAHGLK
ABEGHIKL:EGBAIHLK
ABEGHIJL:EJBAHGLI
ABEGHIJK:EJBAHGIK
ABEFIJKL:EJBAIFLK
ABEFHJKL:EJBFAHLK
ABEFHIKL:EIBFAHLK
ABEFHIJL:EJBFAHLI
ABEFHIJK:EJBFAHIK
ABEFGJKL:EJBFAGLK
ABEFGIKL:EGBAIFLK
ABEFGIJL:EJBFAGLI
ABEFGIJK:EJBFAGIK
ABEFGHKL:EGBFAHLK
ABEFGHJL:HJBFAGLE
ABEFGHJK:HJBFAGEK
ABEFGHIL:EGBFAHLI
ABEFGHIK:EGBFAHIK
ABEFGHIJ:HJBFAGEI
ABDHIJKL:IJBDAHLK
ABDGIJKL:IJBDAGLK
ABDGHJKL:HJBDAGLK
ABDGHIKL:IGBDAHLK
ABDGHIJL:HJBDAGLI
ABDGHIJK:HJBDAGIK
ABDFIJKL:IJBDAFLK
ABDFHJKL:HJBDAFLK
ABDFHIKL:HIBDAFLK
ABDFHIJL:HJBDAFLI
ABDFHIJK:HJBDAFIK
ABDFGJKL:FJBDAGLK
ABDFGIKL:IGBDAFLK
ABDFGIJL:FJBDAGLI
ABDFGIJK:FJBDAGIK
ABDFGHKL:HGBDAFLK
ABDFGHJL:HGBDAFLJ
ABDFGHJK:HGBDAFJK
ABDFGHIL:HGBDAFLI
ABDFGHIK:HGBDAFIK
ABDFGHIJ:HGBDAFIJ
ABDEIJKL:EJBAIDLK
ABDEHJKL:EJBDAHLK
ABDEHIKL:EIBDAHLK
ABDEHIJL:EJBDAHLI
ABDEHIJK:EJBDAHIK
ABDEGJKL:EJBDAGLK
ABDEGIKL:EGBAIDLK
ABDEGIJL:EJBDAGLI
ABDEGIJK:EJBDAGIK
ABDEGHKL:EGBDAHLK
ABDEGHJL:HJBDAGLE
ABDEGHJK:HJBDAGEK
ABDEGHIL:EGBDAHLI
ABDEGHIK:EGBDAHIK
ABDEGHIJ:HJBDAGEI
ABDEFJKL:EJBDAFLK
ABDEFIKL:EIBDAFLK
ABDEFIJL:EJBDAFLI
ABDEFIJK:EJBDAFIK
ABDEFHKL:HEBDAFLK
ABDEFHJL:HJBDAFLE
ABDEFHJK:HJBDAFEK
ABDEFHIL:HEBDAFLI
ABDEFHIK:HEBDAFIK
ABDEFHIJ:HJBDAFEI
ABDEFGKL:EGBDAFLK
ABDEFGJL:EGBDAFLJ
ABDEFGJK:EGBDAFJK
ABDEFGIL:EGBDAFLI
ABDEFGIK:EGBDAFIK
ABDEFGIJ:EGBDAFIJ
ABDEFGHL:HGBDAFLE
ABDEFGHK:HGBDAFEK
ABDEFGHJ:HGBDAFEJ
ABDEFGHI:HGBDAFEI
ABCHIJKL:IJBCAHLK
ABCGIJKL:IJBCAGLK
ABCGHJKL:HJBCAGLK
ABCGHIKL:IGBCAHLK
ABCGHIJL:HJBCAGLI
ABCGHIJK:HJBCAGIK
ABCFIJKL:IJBCAFLK
ABCFHJKL:HJBCAFLK
ABCFHIKL:HIBCAFLK
ABCFHIJL:HJBCAFLI
ABCFHIJK:HJBCAFIK
ABCFGJKL:CJBFAGLK
ABCFGIKL:IGBCAFLK
ABCFGIJL:CJBFAGLI
ABCFGIJK:CJBFAGIK
ABCFGHKL:HGBCAFLK
ABCFGHJL:HGBCAFLJ
ABCFGHJK:HGBCAFJK
ABCFGHIL:HGBCAFLI
ABCFGHIK:HGBCAFIK
ABCFGHIJ:HGBCAFIJ
ABCEIJKL:EJBAICLK
ABCEHJKL:EJBCAHLK
ABCEHIKL:EIBCAHLK
ABCEHIJL:EJBCAHLI
ABCEHIJK:EJBCAHIK
ABCEGJKL:EJBCAGLK
ABCEGIKL:EGBAICLK
ABCEGIJL:EJBCAGLI
ABCEGIJK:EJBCAGIK
ABCEGHKL:EGBCAHLK
ABCEGHJL:HJBCAGLE
ABCEGHJK:HJBCAGEK
ABCEGHIL:EGBCAHLI
ABCEGHIK:EGBCAHIK
ABCEGHIJ:HJBCAGEI
ABCEFJKL:EJBCAFLK
ABCEFIKL:EIBCAFLK
ABCEFIJL:EJBCAFLI
ABCEFIJK:EJBCAFIK
ABCEFHKL:HEBCAFLK
ABCEFHJL:HJBCAFLE
ABCEFHJK:HJBCAFEK
ABCEFHIL:HEBCAFLI
ABCEFHIK:HEBCAFIK
ABCEFHIJ:HJBCAFEI
ABCEFGKL:EGBCAFLK
ABCEFGJL:EGBCAFLJ
ABCEFGJK:EGBCAFJK
ABCEFGIL:EGBCAFLI
ABCEFGIK:EGBCAFIK
ABCEFGIJ:EGBCAFIJ
ABCEFGHL:HGBCAFLE
ABCEFGHK:HGBCAFEK
ABCEFGHJ:HGBCAFEJ
ABCEFGHI:HGBCAFEI
ABCDIJKL:IJBCADLK
ABCDHJKL:HJBCADLK
ABCDHIKL:HIBCADLK
ABCDHIJL:HJBCADLI
ABCDHIJK:HJBCADIK
ABCDGJKL:CJBDAGLK
ABCDGIKL:IGBCADLK
ABCDGIJL:CJBDAGLI
ABCDGIJK:CJBDAGIK
ABCDGHKL:HGBCADLK
ABCDGHJL:HGBCADLJ
ABCDGHJK:HGBCADJK
ABCDGHIL:HGBCADLI
ABCDGHIK:HGBCADIK
ABCDGHIJ:HGBCADIJ
ABCDFJKL:CJBDAFLK
ABCDFIKL:CIBDAFLK
ABCDFIJL:CJBDAFLI
ABCDFIJK:CJBDAFIK
ABCDFHKL:HFBCADLK
ABCDFHJL:CJBDAFLH
ABCDFHJK:HJBCAFDK
ABCDFHIL:HFBCADLI
ABCDFHIK:HFBCADIK
ABCDFHIJ:HJBCAFDI
ABCDFGKL:CGBDAFLK
ABCDFGJL:CGBDAFLJ
ABCDFGJK:CGBDAFJK
ABCDFGIL:CGBDAFLI
ABCDFGIK:CGBDAFIK
ABCDFGIJ:CGBDAFIJ
ABCDFGHL:CGBDAFLH
ABCDFGHK:HGBCAFDK
ABCDFGHJ:HGBCAFDJ
ABCDFGHI:HGBCAFDI
ABCDEJKL:EJBCADLK
ABCDEIKL:EIBCADLK
ABCDEIJL:EJBCADLI
ABCDEIJK:EJBCADIK
ABCDEHKL:HEBCADLK
ABCDEHJL:HJBCADLE
ABCDEHJK:HJBCADEK
ABCDEHIL:HEBCADLI
ABCDEHIK:HEBCADIK
ABCDEHIJ:HJBCADEI
ABCDEGKL:EGBCADLK
ABCDEGJL:EGBCADLJ
ABCDEGJK:EGBCADJK
ABCDEGIL:EGBCADLI
ABCDEGIK:EGBCADIK
ABCDEGIJ:EGBCADIJ
ABCDEGHL:HGBCADLE
ABCDEGHK:HGBCADEK
ABCDEGHJ:HGBCADEJ
ABCDEGHI:HGBCADEI
ABCDEFKL:CEBDAFLK
ABCDEFJL:CJBDAFLE
ABCDEFJK:CJBDAFEK
ABCDEFIL:CEBDAFLI
ABCDEFIK:CEBDAFIK
ABCDEFIJ:CJBDAFEI
ABCDEFHL:HFBCADLE
ABCDEFHK:HEBCAFDK
ABCDEFHJ:HJBCAFDE
ABCDEFHI:HEBCAFDI
ABCDEFGL:CGBDAFLE
ABCDEFGK:CGBDAFEK
ABCDEFGJ:CGBDAFEJ
ABCDEFGI:CGBDAFEI
ABCDEFGH:HGBCAFDE
"""

ANNEX_C_TABLE: Dict[str, Dict[int, str]] = {}
for _line in _ANNEX_C_RAW.strip().splitlines():
    _key, _val = _line.split(":")
    ANNEX_C_TABLE[_key] = dict(zip(ANNEX_C_COLUMN_TO_MATCH, _val))
del _line, _key, _val

# ---------------------------------------------------------------------------
# Bracket tree: Matches 89-102 (Round of 16, Quarter-Finals, Semi-Finals)
#
# Each entry maps a match number to the two earlier match numbers whose
# WINNERS meet in this match. Keys must stay in ascending order so that every
# match's inputs have already been resolved by the time it's processed.
# ---------------------------------------------------------------------------
BRACKET_TREE: Dict[int, Tuple[int, int]] = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
}

FINAL_MATCH = 104        # Winner(101) vs Winner(102)
THIRD_PLACE_MATCH = 103  # Loser(101) vs Loser(102)

# ---------------------------------------------------------------------------
# Bracket layout: which 8 R32 matches feed each half of the draw, in the
# nesting order build_half()/render_half() expects (pairs of adjacent
# entries form one R16 match, pairs of those form one QF match, etc.)
#
#   Left half  -> Semi-Final 101 -> QF 97 = (R16 89, R16 90)
#                                 -> QF 98 = (R16 93, R16 94)
#   Right half -> Semi-Final 102 -> QF 99 = (R16 91, R16 92)
#                                 -> QF 100 = (R16 95, R16 96)
# ---------------------------------------------------------------------------
LEFT_R32_ORDER: List[int] = [74, 77, 73, 75, 83, 84, 81, 82]
RIGHT_R32_ORDER: List[int] = [76, 78, 79, 80, 86, 88, 85, 87]

# Round in which the LOSER of each match is eliminated (used by
# simulate_full_tournament_detailed to populate rounds_reached).
ROUND_OF_LOSER: Dict[int, str] = {}
for _m in R32_MATCHES:
    ROUND_OF_LOSER[_m] = "R32"
for _m in (89, 90, 91, 92, 93, 94, 95, 96):
    ROUND_OF_LOSER[_m] = "R16"
for _m in (97, 98, 99, 100):
    ROUND_OF_LOSER[_m] = "QF"
for _m in (101, 102):
    ROUND_OF_LOSER[_m] = "SF"
ROUND_OF_LOSER[FINAL_MATCH] = "Final"
del _m

# ---------------------------------------------------------------------------
# Match dates / venues (best-effort; cosmetic display only — does not affect
# simulation logic). City names match the conventions used in
# data/teams.py's HOST_NATIONS and data/fixtures.py.
# ---------------------------------------------------------------------------
MATCH_INFO: Dict[int, Dict[str, str]] = {
    73:  {"date": "2026-06-28", "time": "20:00", "venue": "SoFi Stadium", "city": "Los Angeles"},
    74:  {"date": "2026-06-29", "time": "18:00", "venue": "NRG Stadium", "city": "Houston"},
    75:  {"date": "2026-06-29", "time": "21:30", "venue": "Gillette Stadium", "city": "Boston"},
    76:  {"date": "2026-06-30", "time": "02:00", "venue": "Estadio BBVA", "city": "Monterrey"},
    77:  {"date": "2026-06-30", "time": "18:00", "venue": "AT&T Stadium", "city": "Dallas"},
    78:  {"date": "2026-06-30", "time": "22:00", "venue": "MetLife Stadium", "city": "New York/New Jersey"},
    79:  {"date": "2026-07-01", "time": "02:00", "venue": "Estadio Azteca", "city": "Mexico City"},
    80:  {"date": "2026-07-01", "time": "17:00", "venue": "Mercedes-Benz Stadium", "city": "Atlanta"},
    81:  {"date": "2026-07-01", "time": "21:00", "venue": "Lumen Field", "city": "Seattle"},
    82:  {"date": "2026-07-02", "time": "17:00", "venue": "Levi's Stadium", "city": "San Francisco Bay Area"},
    83:  {"date": "2026-07-02", "time": "17:00", "venue": "BC Place", "city": "Vancouver"},
    84:  {"date": "2026-07-02", "time": "21:00", "venue": "BMO Field", "city": "Toronto"},
    85:  {"date": "2026-07-03", "time": "17:00", "venue": "AT&T Stadium", "city": "Dallas"},
    86:  {"date": "2026-07-03", "time": "17:00", "venue": "SoFi Stadium", "city": "Los Angeles"},
    87:  {"date": "2026-07-03", "time": "21:00", "venue": "Hard Rock Stadium", "city": "Miami"},
    88:  {"date": "2026-07-03", "time": "21:00", "venue": "Arrowhead Stadium", "city": "Kansas City"},
    89:  {"date": "2026-07-04", "time": "", "venue": "Lincoln Financial Field", "city": "Philadelphia"},
    90:  {"date": "2026-07-04", "time": "", "venue": "NRG Stadium", "city": "Houston"},
    91:  {"date": "2026-07-05", "time": "", "venue": "MetLife Stadium", "city": "New York/New Jersey"},
    92:  {"date": "2026-07-05", "time": "", "venue": "Estadio Azteca", "city": "Mexico City"},
    93:  {"date": "2026-07-06", "time": "", "venue": "AT&T Stadium", "city": "Dallas"},
    94:  {"date": "2026-07-06", "time": "", "venue": "Lumen Field", "city": "Seattle"},
    95:  {"date": "2026-07-07", "time": "", "venue": "Mercedes-Benz Stadium", "city": "Atlanta"},
    96:  {"date": "2026-07-07", "time": "", "venue": "BC Place", "city": "Vancouver"},
    97:  {"date": "2026-07-09", "time": "", "venue": "Gillette Stadium", "city": "Boston"},
    98:  {"date": "2026-07-10", "time": "", "venue": "SoFi Stadium", "city": "Los Angeles"},
    99:  {"date": "2026-07-11", "time": "", "venue": "Hard Rock Stadium", "city": "Miami"},
    100: {"date": "2026-07-11", "time": "", "venue": "Arrowhead Stadium", "city": "Kansas City"},
    101: {"date": "2026-07-14", "time": "", "venue": "AT&T Stadium", "city": "Dallas"},
    102: {"date": "2026-07-15", "time": "", "venue": "Mercedes-Benz Stadium", "city": "Atlanta"},
    103: {"date": "2026-07-18", "time": "", "venue": "Hard Rock Stadium", "city": "Miami"},
    104: {"date": "2026-07-19", "time": "", "venue": "MetLife Stadium", "city": "New York/New Jersey"},
}


# ---------------------------------------------------------------------------
# Slot resolution
# ---------------------------------------------------------------------------

def describe_slot(slot_code: str) -> str:
    """Human-readable description of an R32 slot code, for tooltips."""
    kind = slot_code[0]
    if kind == "1":
        return f"Winner Group {slot_code[1]}"
    if kind == "2":
        return f"Runner-up Group {slot_code[1]}"
    if slot_code.startswith("3:"):
        groups = "/".join(slot_code[2:].split(","))
        return f"Best 3rd Place ({groups})"
    return slot_code


def resolve_third_place_assignment(qualifying_groups: List[str]) -> Dict[int, str]:
    """
    Given the 8 group letters whose 3rd-placed team qualifies for the
    knockout stage, return {match_number: group_letter} mapping each of the
    8 "best 3rd place" R32 slots (THIRD_PLACE_SLOTS) to the group whose
    3rd-placed team fills it.

    Literal lookup into FIFA's published Annex C table (ANNEX_C_TABLE) --
    see module docstring. Raises ValueError if the input isn't exactly 8
    distinct group letters or has no Annex C entry (i.e. isn't a valid
    8-of-12 combination of A-L).
    """
    groups = sorted(qualifying_groups)
    if len(groups) != 8 or len(set(groups)) != 8:
        raise ValueError(f"Expected 8 distinct qualifying third-place groups, got {groups}")

    key = "".join(groups)
    try:
        return dict(ANNEX_C_TABLE[key])
    except KeyError:
        raise ValueError(f"No Annex C entry for qualifying groups {groups}") from None


def _resolve_slot(
    slot_code: str,
    match_num: int,
    winners: Dict[str, str],
    runnersup: Dict[str, str],
    thirds: Dict[str, str],
    third_assignment: Dict[int, str],
) -> str:
    kind = slot_code[0]
    if kind == "1":
        return winners[slot_code[1]]
    if kind == "2":
        return runnersup[slot_code[1]]
    if slot_code.startswith("3:"):
        group = third_assignment[match_num]
        return thirds[group]
    raise ValueError(f"Unrecognised slot code: {slot_code!r}")


def resolve_r32_pairing(
    winners: Dict[str, str],
    runnersup: Dict[str, str],
    thirds: Dict[str, str],
    third_assignment: Dict[int, str],
) -> Dict[int, Tuple[str, str]]:
    """
    Resolve all 16 Round-of-32 matchups (matches 73-88) to actual team names.

    winners / runnersup / thirds: {group_letter: team_name}
    third_assignment: from resolve_third_place_assignment()
    """
    return {
        m: (
            _resolve_slot(slot_a, m, winners, runnersup, thirds, third_assignment),
            _resolve_slot(slot_b, m, winners, runnersup, thirds, third_assignment),
        )
        for m, (slot_a, slot_b) in R32_MATCHES.items()
    }


def walk_bracket(
    r32_pairing: Dict[int, Tuple[str, str]],
    decide_fn: Callable[[str, str, int], str],
) -> Dict[int, Tuple[str, str]]:
    """
    Walk the full knockout bracket (matches 73-104) given resolved R32
    pairings and a decision function.

    decide_fn(team_a, team_b, match_num) -> winning team name

    Returns {match_number: (winner, loser)} for all 32 matches 73-104,
    including the 3rd-place match (103) and Final (104).
    """
    results: Dict[int, Tuple[str, str]] = {}

    for m, (a, b) in r32_pairing.items():
        w = decide_fn(a, b, m)
        results[m] = (w, b if w == a else a)

    for m, (ma, mb) in BRACKET_TREE.items():
        a, b = results[ma][0], results[mb][0]
        w = decide_fn(a, b, m)
        results[m] = (w, b if w == a else a)

    a, b = results[101][0], results[102][0]
    w = decide_fn(a, b, FINAL_MATCH)
    results[FINAL_MATCH] = (w, b if w == a else a)

    la, lb = results[101][1], results[102][1]
    w3 = decide_fn(la, lb, THIRD_PLACE_MATCH)
    results[THIRD_PLACE_MATCH] = (w3, lb if w3 == la else la)

    return results
