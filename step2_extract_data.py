import spacy
import pandas as pd
import re

# Load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')


# Define a function to extract relevant information from an expert profile
def extract_data(profile):
    doc = nlp(profile)

    # Extract education level
    education_levels = {
        'kỹ sư': 1,
        'thạc sỹ': 2,
        'tiến sỹ': 3,
        'giáo sư': 4
    }
    education = 0
    # for token in doc:
    pattern = r"\b(giáo sư|tiến sỹ|thạc sỹ)\b"
    match = re.search(pattern, profile)
    if match:
        if match.group() in education_levels:
            education = education_levels[match.group()]

    # Extract years of experience
    pattern = r"\d+\s*năm kinh nghiệm"
    match = re.search(pattern, profile)
    years_of_exp = 0
    if match:
        result = re.search(r'\d+', match.group(0))
        if result:
            years_of_exp = int(result.group())

    # Extract number of papers
    pattern =  r"\d+\s*(bài báo|nghiên cứu)"
    match = re.search(pattern, profile)
    papers = 0
    if match:
        papers = int(match.group(0).split()[0])

    # Extract number of awards received
    pattern = r"\d+\s*(giải thưởng|phần thưởng)"
    match = re.search(pattern, profile)
    awards = 0
    if match:
        awards = int(match.group(0).split()[0])

    return (education, years_of_exp, papers, awards)


# Load profiles from csv file
df = pd.read_csv('step2_data.csv')

# Extract data from the expert profiles and store in a Pandas DataFrame
data = []
for profile in df['profile']:
    result = extract_data(profile)
    data.append(result)
pd.set_option('display.max_columns', None)
df = pd.DataFrame(data, columns=['education_level', 'years_of_experience', 'papers', 'awards'])

# Print the merged DataFrame
print(df)
