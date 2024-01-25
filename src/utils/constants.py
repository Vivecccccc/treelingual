INST_TEMPLATE_SEQ = """
Please convert the query into a sequence of functions. The sequence can be executed in a specifically designed engine to find the answer to a query. When you see any proper nouns, please keep it unchanged in the output.

Example 1:
Input:
Is the name of the person that was educated at high school Laura Linney?
Output:
Find <arg> high school <func> Relate <arg> educated at <arg> backward <func> FilterConcept <arg> human <func> QueryAttr <arg> name in native language <func> VerifyStr <arg> Laura Linney

Example 2:
Input:
How many high schools are there whose country is the sovereign state that has Germany-Guinea diplomatic relations with Germany?
Output:
Find <arg> Germany <func> Relate <arg> diplomatic relation <arg> backward <func> QFilterStr <arg> statement is subject of <arg> Germany-Guinea relations <func> FilterConcept <arg> sovereign state <func> Relate <arg> country <arg> backward <func> FilterConcept <arg> high school <func> Count

Example 3:
Input:
When was the instance of National Football League Draft whose English language website is http://www.nfl.com/draft/history/fulldraft?season=2007?
Output:
FindAll <func> FilterStr <arg> official website <arg> http://www.nfl.com/draft/history/fulldraft?season=2007 <func> QFilterStr <arg> language of work or name <arg> English <func> FilterConcept <arg> National Football League Draft <func> QueryAttr <arg> point in time
"""

INST_TEMPLATE_TREE_SIMPLE = """
Please convert the query into an XML-like tree of functions. The functions and the XML-like tree structure can be executed in a specifically designed engine to find the answer to a query. When you see any proper nouns, please keep it unchanged in the output; Please remember to put arguments (if any) inside of the tag as an attribute, separated by ;;.

Example 1:
Input:
Is the name of the person that was educated at high school Laura Linney?
Output:
<VerifyStr args="Laura Linney"><QueryAttr args="name in native language"><FilterConcept args="human"><Relate args="educated at;;backward"><Find args="high school" /></Relate></FilterConcept></QueryAttr></VerifyStr>

Example 2:
Input:
How many high schools are there whose country is the sovereign state that has Germany-Guinea diplomatic relations with Germany?
Output:
<Count><FilterConcept args="high school"><Relate args="country;;backward"><FilterConcept args="sovereign state"><QFilterStr args="statement is subject of;;Germany-Guinea relations"><Relate args="diplomatic relation;;backward"><Find args="Germany" /></Relate></QFilterStr></FilterConcept></Relate></FilterConcept></Count>

Example 3:
Input:
When was the instance of National Football League Draft whose English language website is http://www.nfl.com/draft/history/fulldraft?season=2007?
Output:
<QueryAttr args="point in time"><FilterConcept args="National Football League Draft"><QFilterStr args="language of work or name;;English"><FilterStr args="official website;;http://www.nfl.com/draft/history/fulldraft?season=2007"><FindAll /></FilterStr></QFilterStr></FilterConcept></QueryAttr>
"""

INST_TEMPLATE_TREE_COMPLEX = """
Please convert the query into an XML-like tree of functions. The functions and the XML-like tree structure can be executed in a specifically designed engine to find the answer to a query.

Example 1:
Input:
Is the name of the person that was educated at high school Laura Linney?
Output:
<VerifyStr><QueryAttr><FilterConcept><Relate><Find><args><arg>high school</arg></args></Find><args><arg>educated at</arg><arg>backward</arg></args></Relate><args><arg>human</arg></args></FilterConcept><args><arg>name in native language</arg></args></QueryAttr><args><arg>Laura Linney</arg></args></VerifyStr>

Example 2:
Input:
How many high schools are there whose country is the sovereign state that has Germany-Guinea diplomatic relations with Germany?
Output:
<Count><FilterConcept><Relate><FilterConcept><QFilterStr><Relate><Find><args><arg>Germany</arg></args></Find><args><arg>diplomatic relation</arg><arg>backward</arg></args></Relate><args><arg>statement is subject of</arg><arg>Germany-Guinea relations</arg></args></QFilterStr><args><arg>sovereign state</arg></args></FilterConcept><args><arg>country</arg><arg>backward</arg></args></Relate><args><arg>high school</arg></args></FilterConcept></Count>

Example 3:
Input:
When was the instance of National Football League Draft whose English language website is http://www.nfl.com/draft/history/fulldraft?season=2007?
Output:
<QueryAttr><FilterConcept><QFilterStr><FilterStr><FindAll></FindAll><args><arg>official website</arg><arg>http://www.nfl.com/draft/history/fulldraft?season=2007</arg></args></FilterStr><args><arg>language of work or name</arg><arg>English</arg></args></QFilterStr><args><arg>National Football League Draft</arg></args></FilterConcept><args><arg>point in time</arg></args></QueryAttr>
"""
# INST_TEMPLATE_TREE_COMPLEX = """
# Please convert the query into an XML-like tree of functions. The functions and the XML-like tree structure can be executed in a specifically designed engine to find the answer to a query.

# Example 1:
# Input:
# Is http://www.cheechandchong.com Eve Myles's official website?
# Output:
# <VerifyStr><QueryAttr><Find><args><arg>Eve Myles</arg></args></Find><args><arg>official website</arg></args></QueryAttr><args><arg>http://www.cheechandchong.com</arg></args></VerifyStr>

# Example 2:
# Input:
# How many neighborhoods have a population not 500000000?
# Output:
# <Count><FilterConcept><FilterNum><FindAll></FindAll><args><arg>population</arg><arg>500000000</arg><arg>!=</arg></args></FilterNum><args><arg>neighborhood</arg></args></FilterConcept></Count>
# """

ARG_TOKENS = ["<args>", "</args>", "<arg>", "</arg>"]
FUNCTION_LIST = [
    "And",
    "Count",
    "FilterConcept",
    "FilterDate",
    "FilterNum",
    "FilterStr",
    "FilterYear",
    "Find",
    "FindAll",
    "Or",
    "QFilterDate",
    "QFilterNum",
    "QFilterStr",
    "QFilterYear",
    "QueryAttr",
    "QueryAttrQualifier",
    "QueryAttrUnderCondition",
    "QueryName",
    "QueryRelation",
    "QueryRelationQualifier",
    "Relate",
    "SelectAmong",
    "SelectBetween",
    "VerifyDate",
    "VerifyNum",
    "VerifyStr",
    "VerifyYear"
]
FUNCTION_MAP = {x: {"start": f"<{x}>", "end": f"</{x}>"} for _, x in enumerate(FUNCTION_LIST)}