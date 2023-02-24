import requests

from bs4 import BeautifulSoup 
import os 

cwd = os.getcwd()

file = open('raw_authorlist.txt')
output_file_name = "links.txt"

all_authors =[]
urls = []
counter = 0  
no_profile_counter = 0

author_paper_dict = {} 

print(len(file), ": The number of authors in given list.")

with open(output_file_name, 'a+') as f:
    for line in file:  
        og_author_string = line.strip() 
        
        author_name = line.strip().split()
        all_authors.append(line.strip())
        author_string = all_authors[-1].replace(' ', '+') 
        #gets 4 papers for the given author
        get_req_string = "https://api.semanticscholar.org/graph/v1/author/search?query=" + author_string + "&fields=name,aliases,url,papers.title,papers.year&limit=1"
        response = requests.get(get_req_string)
        if (response.status_code == 200): 
            response_json = response.json()
            data = response_json['data']
            if (len(data) != 0): 
                author_paper_dict[og_author_string] = []
                curr_url = data[0]['url'] 
                author_id = data[0]['authorId']
                get_paper_string = "https://api.semanticscholar.org/graph/v1/author/" + author_id + "/papers?fields=url,year,authors&limit=4"
                response2 = requests.get(get_paper_string) 
                paper_ids = [item['paperId'] for item in response2.json()['data']] 
                f.write(og_author_string + " \n") 
                paper_urls = []
                for id in paper_ids: 
                    string = "https://api.semanticscholar.org/graph/v1/paper/" + id 
                    string+= "?fields=" + "url,isOpenAccess,openAccessPdf" 
                    r = requests.get(string)   
                    paper_urls.append(r.json()['url'])
                    is_open = r.json()['isOpenAccess']
                    pdf_link = (r.json()['openAccessPdf'])
                    if is_open: 
                        author_paper_dict[og_author_string].append(pdf_link['url']) 
                        f.write(pdf_link['url'] + "\n") 
                f.write("\n")
                print(counter)
                counter += 1   
                i = 0
                for lnk in paper_urls: 
                    i += 1
                                # Requests URL and get response object
                    response = requests.get(lnk)
            
                    # Parse text obtained
                    soup = BeautifulSoup(response.text, 'html.parser')
            
                    # Find all hyperlinks present on webpage
                    links = soup.find_all('a')

                    # From all links check for pdf link and
                    # if present download file
                    for link in links:
                        if ('.pdf' in link.get('href', [])): 

                            print(link.get("href"))
                            print("Downloading file: ", i)
                            # Get response object for link
                            pdf_response = requests.get(link.get('href'))
                            # Write content in pdf file
                            os.chdir(cwd + "/authors-test")
                            pdf = open(og_author_string + str(i)+".pdf", 'wb')
                            pdf.write(pdf_response.content)
                            pdf.close()
                            print(og_author_string + " File ", i, " downloaded")
            else: 
                print(author_name[0] + author_name[1]  + " does not have a semantic scholar profile")
                no_profile_counter += 1 




print(str(no_profile_counter) + ": number of authors that did not have semantic scholar profiles")


# author_string = "nihar+shah" 
# get_req_string = "https://api.semanticscholar.org/graph/v1/author/search?query=" + author_string + "&fields=name,aliases,url,papers.title,papers.year&limit=1"
# response = requests.get(get_req_string)
# print(response)
# author_id = response.json()['data'][0]['authorId']
# get_paper_string = "https://api.semanticscholar.org/graph/v1/author/" + author_id + "/papers?fields=url,year,authors&limit=2"

# response2 = requests.get(get_paper_string) 
# paper_ids = [item['paperId'] for item in response2.json()['data']] 

# print(paper_ids)

# for id in paper_ids: 
#     string = "https://api.semanticscholar.org/graph/v1/paper/" + id 
#     string+= "?fields=" + "isOpenAccess,openAccessPdf" 
#     r = requests.get(string) 
#     print(r.json())