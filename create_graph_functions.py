from pyvis.network import Network
def create_graph(edge_dict, question, output_file="graph.html", report_file="report.html", client=None):
    """
    Creates a Pyvis network graph from a dictionary where:
      - Each key is a tuple (node1, node2) representing an edge.
      - Each value is a textual description for that edge.
      
    The edge description is only shown as a tooltip (on hover/click). After creating
    the graph, a report is generated using a chat completion API and saved to a separate HTML file.
    
    Parameters:
      edge_dict (dict): Dictionary with keys as tuples (node1, node2) and values as edge descriptions.
      output_file (str): Filename for the output HTML graph.
      report_file (str): Filename for the output HTML report.
      client: An initialized client for chat completions (e.g. OpenAI client or similar).
    """
    # Initialize Pyvis Network; adjust settings as needed.
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=False)
    
    # Gather all unique nodes.
    nodes = set()
    for (node_a, node_b) in edge_dict.keys():
        nodes.add(node_a)
        nodes.add(node_b)
    
    # Add nodes to the network.
    for node in nodes:
        net.add_node(node, label=node)
    
    # Add edges with the description set as a tooltip only.
    for (node_a, node_b), description in edge_dict.items():
        net.add_edge(node_a, node_b, title=description)
    
    # Generate the network graph as an HTML file.
    net.show(output_file, notebook=False)
    print(f"Graph has been created and saved to {output_file}")
    
    # Generate a report using the chat completions API.
    # Make sure 'client' is provided and configured appropriately.
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             """Write a rigorous report based on the research findings stored in dictionary format. 
Include the original question as the focus of the report.The original question is: """ + question},
            {"role": "user", "content": str(edge_dict)}
        ]
    )

    response = completion.choices[0].message.content

    # Write the report to an HTML file.
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Research Report</title>
    <style>
      body {{
         font-family: Arial, sans-serif;
         margin: 20px;
      }}
      pre {{
         background-color: #f4f4f4;
         padding: 15px;
         border-radius: 5px;
         white-space: pre-wrap;
      }}
    </style>
</head>
<body>
    <h1>Research Report</h1>
    <pre>{response}</pre>
</body>
</html>"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report has been created and saved to {report_file}")