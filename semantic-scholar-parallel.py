import requests

from bs4 import BeautifulSoup 
import os 
import concurrent.futures 

cwd = os.getcwd()

file = open('neurips-authorlist.txt ')
output_file_name = "links.txt"

all_authors =[]
no_profile_counter = 0 

author_to_num_papers = {}

author_paper_dict = {} 

"""
The given algorithm extracts 4 most relvant papers listed on an authors profile on semantic scholar with high likelihood. There are cases where the number of papers downlaoded is < 4: 
- the author does not have a semantic scholar profile 
- a majority of links are not accessible/give exceptions when sent a get request. 
"""





def process_line(line, no_profile_counter, author_to_num_papers,author_paper_dict): 
    og_author_string = line.strip() 
    author_paper_count = 0
    author_string = og_author_string.replace(' ', '+') 
    #gets 4 papers for the given author
    get_req_string = "https://api.semanticscholar.org/graph/v1/author/search?query=" + author_string + "&fields=name,aliases,url,papers.title,papers.year&limit=1"
    response = requests.get(get_req_string)
    if (response.status_code == 200): 
        response_json = response.json()
        data = response_json['data'] 
        if (og_author_string == "Aaron Schein"): 
            print(data)
        if (len(data) != 0): 
            author_paper_dict[og_author_string] = []
            curr_url = data[0]['url'] 
            author_id = data[0]['authorId']
            get_paper_string = "https://api.semanticscholar.org/graph/v1/author/" + author_id + "/papers?fields=url,year,authors&limit=15"
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
            for lnk in paper_urls: 
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
                        if (author_paper_count >= 4): 
                            break 
                        author_paper_count += 1
                        print(link.get("href"))
                        print("Downloading file: ", author_paper_count)
                        # Get response object for link
                        try:
                            pdf_response = requests.get(link.get('href'))
                        except requests.exceptions.RequestException as e: #handles all exceptions inclduing timeout, http, connectionError etc. 
                            author_paper_count -= 1 
                            continue 
                        pdf_response = requests.get(link.get('href'))
                        # Write content in pdf file
                        os.chdir(cwd + "/authors-test")
                        pdf = open(og_author_string + str(author_paper_count)+".pdf", 'wb')
                        pdf.write(pdf_response.content)
                        pdf.close()
                        print(og_author_string + " File ", author_paper_count, " downloaded")
                if (author_paper_count >= 4): 
                    break 

            author_to_num_papers[og_author_string] = author_to_num_papers
        else: 
            print(og_author_string  + " does not have a semantic scholar profile")
            no_profile_counter += 1 


counter = 0

if __name__ == '__main__': 

    def _count_generator(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    with open(r'neurips-authorlist.txt ', 'rb')as fp:
        c_generator = _count_generator(fp.raw.read)
        # count each \n
        count = sum(buffer.count(b'\n') for buffer in c_generator)
        print('Total lines:', count + 1) 

    with open(output_file_name, 'a+') as f: 

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(process_line, line, no_profile_counter, author_to_num_papers,author_paper_dict) for line in file]

        for future in concurrent.futures.as_completed(results):
            continue 

    print(str(no_profile_counter) + ": number of authors that did not have semantic scholar profiles") 


