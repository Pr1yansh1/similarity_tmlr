import requests 
import os  
import glob 

os.chdir("/Users/priyanshigarg/Desktop/submissions/similarity_tmlr/submission-pdfs")
# Get a list of all filenames in the current directory using glob
file_paths = glob.glob('*')


# Sort the file paths in the same order as returned by glob
papers = sorted(file_paths, key=lambda x: glob.glob(x))
file_paths = [(item.split("."))[0] for item in papers]
url = 'https://api2.openreview.net/notes?invitation=TMLR/-/Submission'
df = requests.get(url).json()['notes'] 

#mapping paper-ids to cdates 

papers_dict = {}
index = 0 
for p in papers: 
    papers_dict[index] = None 
    index += 1

current_papers = [ ]
for item in df: 
    id = str(item["id"])
    current_papers.append(id)
    c = item["cdate"] 
    title = item["content"]["title"]["value"]
    if (id in file_paths): 
        index = file_paths.index(id)
        papers_dict[index] = c 

list1 = current_papers 
list2 = papers


print(papers_dict)