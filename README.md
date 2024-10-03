## Anime Recommendation System Documentation

This system uses sentence embeddings to power a content-based anime recommendation engine. Given a new description of an anime, the system finds similar anime within a data set of existing anime.

### Inputs

The system requires the following inputs:  

*   A new anime description.  
*   A pre-trained sentence embedding model, such as `all-mpnet-base-v2`.

### Outputs

The system returns a ranked list of the most similar anime in the dataset, along with similarity scores. 
