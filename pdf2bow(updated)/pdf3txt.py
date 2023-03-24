import PyPDF2
import argparse
import os
import re
from glob import glob


def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0 


def fix_pdf(name): 
    print(name)
    def reset_eof_of_pdf_return_stream(pdf_stream_in:list):
        # find the line position of the EOF

        actual_line = 0
        for i, x in enumerate(txt[::-1]):
            if b'%%EOF' in x:
                actual_line = len(pdf_stream_in)-i
                print(f'EOF found at line position {-i} = actual {actual_line}, with value {x}')
                break

        # return the list up to that point
        return pdf_stream_in[:actual_line]

    # opens the file for reading
    with open("input/" + name + ".pdf", 'rb') as p:
        txt = (p.readlines())

    # get the new list terminating correctly
    txtx = reset_eof_of_pdf_return_stream(txt)

    # write to new pdf
    with open("fixed-input/" + name + '_fixed.pdf', 'wb') as f:
        f.writelines(txtx)


def write(name):

    #create file object variable
    #opening method will be rb

    path = "fixed-input/" + name + '_fixed.pdf' 

    if is_non_zero_file(path):
        pdffileobj=open("fixed-input/" + name + '_fixed.pdf','rb')
        
        #create reader variable that will read the pdffileobj
        pdfreader=PyPDF2.PdfReader(pdffileobj)
        
        #This will store the number of pages of this pdf file
        x= len(pdfreader.pages)
        
        #create a variable that will select the selected number of pages
        pageobj=pdfreader.pages[x-1]
        
        #(x+1) because python indentation starts with 0.
        #create text variable which will store all text datafrom pdf file
        text=pageobj.extract_text()
        
        #save the extracted data from pdf to a txt file
        #we will use file handling here
        #dont forget to put r before you put the file path
        #go to the file location copy the path by right clicking on the file
        #click properties and copy the location path and paste it here.
        #put "\\your_txtfilename"
        file1=open(r"output/" + name + ".txt","a")
        file1.writelines(text)


def parse_args():
    parser = argparse.ArgumentParser(description='pdf2bow')
    parser.add_argument('--output_dir', type=str, required=False, default='.', help="output directory")
    parser.add_argument('--input', type=str, required=True, help="input PDF or directory")
    parser.add_argument('--overwrite', type=bool, required=False,
                        help="whether or not to re-process previously process PDFs", default=False)

    args = parser.parse_args()

    return args

def modify(item): 
    n = item[6:] 
    l = len(n) 
    n = n[: l - 4 ]
    return n 

def run(): 
    args = parse_args()
    if os.path.isdir(args.input):
        print('Parsing all pdfs in the directory', args.input)
        g = glob(args.input + '/*') 
        names = [modify(item) for item in g]

        for name in names:
            fix_pdf(name)
            write(name)

if __name__ == '__main__':
    run()
 


