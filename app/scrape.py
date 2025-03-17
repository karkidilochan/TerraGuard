import requests
import frontmatter
import re

TF_GIT_URL = "https://api.github.com/repos/hashicorp/terraform-provider-aws/contents/website/docs/r"


def get_resource(file):
    file_response = requests.get(file["download_url"])
    doc = file_response.text
    resource = frontmatter.loads(doc)
    return resource


def extract_metadata(resource):
    metadata = dict(resource.metadata)

    # Extract main content
    main_content = resource.content

    # Separate sections
    sections = {}
    current_section = "main"
    sections[current_section] = []

    for line in main_content.split("\n"):
        if line.startswith("## "):
            current_section = line.replace("## ", "").strip()
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    structured_content = {
        "title": metadata.get("page_title", ""),
        "description": metadata.get("description", ""),
        "resource_name": (
            re.search(r"# Resource: (.*)", main_content).group(1)
            if re.search(r"# Resource: (.*)", main_content)
            else ""
        ),
        "frontmatter": metadata,
        "sections": {k: "\n".join(v) for k, v in sections.items()},
        "full_content": main_content,
    }


def get_resources():
    response = requests.get(TF_GIT_URL)
    doc_files = response.json()
    resources = []
    for file in doc_files:
        if file["name"].endswith(".markdown"):
            raw_resource = get_resource(file)

            resources.append()
    return resources
