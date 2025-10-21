import requests
import xml.etree.ElementTree as ET

ns = {
    'dc': 'http://purl.org/dc/elements/1.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'cinii': 'http://ci.nii.ac.jp/ns/1.0/',
    'default': 'https://cir.nii.ac.jp/schema/1.0/'
}


# Load XML
def load_xml(xml_text):
    return ET.fromstring(xml_text)


# Extract title
def extract_title(root):
    title_elem = root.find('.//dc:title', ns)
    return title_elem.text if title_elem is not None else None


# Extract abstract
def extract_abstract(root):
    # Find description element whose type is Abstract
    for desc in root.findall('.//default:description', ns):
        type_elem = desc.find('default:type', ns)
        if type_elem is not None and type_elem.text == 'Abstract':
            notation_elem = desc.find('default:notation', ns)
            if notation_elem is not None:
                return notation_elem.text
    return None


# Extract keywords (subjects)
def extract_keywords(root):
    keywords = []
    # Extract all notation under dcterms:subject
    for subject in root.findall('.//dcterms:subject', ns):
        for notation in subject.findall('.//*'):
            if notation.tag.endswith('notation') and notation.text:
                keywords.append(notation.text)
    return keywords


# Example usage
if __name__ == "__main__":
    uri = "https://cir.nii.ac.jp/crid/1050574411837070592.rdf"

    r = requests.get(uri)
    r.raise_for_status()
    xml_text = r.text
    print(xml_text)

    root = load_xml(xml_text)
    title = extract_title(root)
    abstract = extract_abstract(root)
    keywords = extract_keywords(root)

    print("Title:", title)
    print("Abstract:", abstract)
    print("Keywords:", keywords)
