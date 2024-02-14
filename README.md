## Machine Reading Comprehension

### Dataset 

```
from datasets import Dataset

path = 'dataset.arrow'
data = Dataset.from_file(path)
data
# Dataset({
#     features: ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__'],
#     num_rows: 3952
# })
