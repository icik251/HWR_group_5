## Basic setup for pipeline:
Line segmentation > character segmentation > character recognition > 
character style classification > image style classification

### Line Segmentation
Input: image (jpeg)<br>
Output: lines (jpeg)

### Character segmentation
Input: lines (jpeg)<br>
Output: characters (jpeg)

### Character recognition
Input: characters (jpeg)<br>
Output: characters (jpeg) + labels (txt)

### Character style classification
Input: characters (jpeg) + labels (txt)<br>
Output: character style labels (txt)

### Image style classification
Input: character style labels (txt)<br>
=======
## Basic setup for pipeline:
Line segmentation > character segmentation > character recognition > 
character style classification > image style classification

### Line Segmentation
Input: image (jpeg)<br>
Output: lines (jpeg)

### Character segmentation
Input: lines (jpeg)<br>
Output: characters (jpeg)

### Character recognition
Input: characters (jpeg)<br>
Output: characters (jpeg) + labels (txt)

### Character style classification
Input: characters (jpeg) + labels (txt)<br>
Output: character style labels (txt)

### Image style classification
Input: character style labels (txt)<br>
Output: image style label (txt)

## Folders
### Images
Output Images

### src
Source code

### src/character_segmentation
Code regarding character segmentation

### src/character_recognition
Code regarding character recognition

### src/style_classification
Code regarding style classification

