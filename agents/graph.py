from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.search_node import search_node
from agents.summary_node import summary_node
from agents.validation_node import validation_node
from utils.exporters import export_to_markdown


class GraphState(TypedDict, total=False):
    """The State schema for the entire research process."""

    query: str
    vector_store: Any
    local_chunks: list
    web_snippet: str
    report_dict: dict
    final_report_md: str
    generation_warning: str


def query_node(state):
    """Dummy initialization node to trigger the rest of the flow."""
    return state


def report_node(state):
    """Formats the generated report dictionary into a markdown format string."""
    report_dict = state.get("report_dict", {})
    md_content = export_to_markdown(report_dict)
    state["final_report_md"] = md_content
    return state


def build_research_graph():
    """Builds and wires together the LangGraph workflow pipeline."""
    workflow = StateGraph(GraphState)

    workflow.add_node("query_node", query_node)
    workflow.add_node("search_node", search_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("summary_node", summary_node)
    workflow.add_node("report_node", report_node)

    workflow.add_edge(START, "query_node")
    workflow.add_edge("query_node", "search_node")
    workflow.add_edge("search_node", "validation_node")
    workflow.add_edge("validation_node", "summary_node")
    workflow.add_edge("summary_node", "report_node")
    workflow.add_edge("report_node", END)

    return workflow.compile()


def run_research_agent(query, vector_store):
    """Run the graph and return the final state."""
    app = build_research_graph()

    initial_state = {
        "query": query,
        "vector_store": vector_store,
        "local_chunks": [],
        "web_snippet": "",
        "report_dict": {},
        "final_report_md": "",
        "generation_warning": "",
    }

    final_state = app.invoke(initial_state)
    return final_state
