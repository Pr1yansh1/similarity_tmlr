# Similarity Computation 


Run command: 

```
python3 build_dict.py 
```

authors-test: 

folder containing .txt files of all the authors. File names should be formatted like: 

``` 
"{Author Name}.{info}.txt" 
``` 

papers-test: 

folder containing .txt files of all the papers of which similarity will be computed 

resulting matrix of size #num of reviewers * #num_papers 

The matrix in 'similarity_result.txt' has 419 rows and 811 columns.



The given algorithm extracts 4 most relvant papers listed on an authors profile on semantic scholar with high likelihood. There are cases where the number of papers downlaoded is < 4: 
- the author does not have a semantic scholar profile 
- a majority of links are not accessible/give exceptions when sent a get request. 

dependencies: 

python packages: 

requests, beautifulsoup4, 
