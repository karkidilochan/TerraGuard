import requests
import frontmatter
import re
import json
from langchain_core.documents import Document
from typing import List
import uuid
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter

TF_GIT_URL = "https://api.github.com/repos/hashicorp/terraform-provider-aws/contents/website/docs/r"


def get_resource(file):
    file_response = requests.get(file["download_url"])
    doc = file_response.text
    print(f"Getting resource: {file['name']}")
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
        "metadata": metadata,
        "sections": {k: "\n".join(v) for k, v in sections.items()},
        "full_content": main_content,
    }

    return structured_content


def get_resources():
    response = requests.get(TF_GIT_URL)
    doc_files = response.json()
    resources = []
    for file in tqdm(doc_files, desc="Processing markdown files"):
        if file["name"].endswith(".markdown"):
            raw_resource = get_resource(file)
            structured_resource = extract_metadata(raw_resource)
            resources.append(structured_resource)
    with open("resources.json", "w") as json_file:
        json.dump(resources, json_file, indent=4)
    return resources


def chunk_aws_resources() -> List[Document]:
    chunks = []

    with open("aws_resources.json", "r") as json_file:
        resources = json.load(json_file)
        for resource in tqdm(resources, desc="Processing resources:"):
            title = resource["title"]
            description = resource["description"]
            resource_name = resource["resource_name"]
            sub_category = resource["metadata"]["subcategory"]
            sections = resource["sections"]
            for section_name, content in sections.items():
                # chunks.append(
                #     {
                #         "metadata": {
                #             "title": title,
                #             "description": description,
                #             "resource_name": resource_name,
                #             "metadata": metadata,
                #             "section": section_name,
                #             "type": "code" if "```terraform" in content else "text",
                #         },
                #         "page_content": content,
                #     }
                # )
                chunks.append(
                    Document(
                        # id=str(uuid.uuid4()),
                        metadata={
                            "title": title,
                            "description": description,
                            "resource_name": resource_name,
                            "subcategory": sub_category,
                            "section": section_name,
                            "type": "code" if "```terraform" in content else "text",
                        },
                        page_content=content,
                    )
                )

        return chunks


if __name__ == "__main__":
    get_resources()
