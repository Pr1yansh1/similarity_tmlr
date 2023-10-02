import pandas as pd
import requests

url = 'https://api2.openreview.net/notes?invitation=TMLR/-/Submission'
df = pd.DataFrame(requests.get(url).json()['notes'])

for note_id in df['id']:
  response = requests.get('https://api2.openreview.net/pdf?id='+note_id)
  pdf = open(note_id + ".pdf", 'wb')
  pdf.write(response.content)
  pdf.close()
  print("File "+note_id + " downloaded")
