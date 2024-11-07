# CV ANALYSIS AND CANDIDATE RANKING TOOL 📄🏆

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.13.0-lightgrey.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Transformers-v4.31.0-orange.svg)](https://huggingface.co/transformers/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.6.3-brightgreen.svg)](https://matplotlib.org/)
[![Torch](https://img.shields.io/badge/Torch-v1.13.0-blue.svg)](https://pytorch.org/)

## Overview

The **CV ANALYSIS AND CANDIDATE RANKING TOOL** allows users to upload and analyze CVs in both **PDF** and **Word** formats. The application leverages the **DistilBERT model** from Hugging Face to assess the skills and experience of candidates, generating a ranking based on the analysis of their CVs.

## Features

- Supports both **PDF** and **Word** file uploads for CV analysis
- **DistilBERT** model used for analyzing skills and experience from CV text
- Scores are calculated for skills and experience, generating a ranking for candidates
- **Streamlit** interface for a user-friendly experience
- Data visualization with **Matplotlib** to compare candidates' scores

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://huggingface.co/spaces/KhaqanNasir/cv-analysis
   
2. Navigate to the project directory:

   ```bash
   cd cv-analysis
  
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   
## Usage
1. Run the application:

   ```bash
   streamlit run app.py
   
2. Upload PDF or Word CV files and wait for the text extraction and analysis to complete.

3. View the candidate ranking, detailed scores, and comparison of skills and experience.

## Contributing
Contributions are welcome! To contribute to the project, please follow these steps:

1. Fork the repository.

    ```bash
    git clone https://huggingface.co/spaces/KhaqanNasir/cv-analysis  

2. Create a new branch:

   ```bash
   git checkout -b feature/YourFeatureName
   
3. Make your changes and commit them:

   ```bash
   git commit -m "Add your message here"
   
4. Push to your branch:

   ```bash
   git push origin feature/YourFeatureName
   
5. Create a pull request.

## License
This project is licensed under the MIT License. See the <a href="https://github.com/KhaqanNasir/cv-analysis/blob/main/LICENSE">LICENSE</a> file for details.

## Acknowledgments
Developed by <a href="https://www.linkedin.com/in/khaqan-nasir/">Muhammad Khaqan Nasir</a><br> Inspired by the need to streamline the hiring process and evaluate candidate qualifications through automated CV analysis.

## Configuration Reference
For additional configuration options, check out the configuration reference.

## Additional Notes:
1. Dependencies: Ensure you create a requirements.txt file that includes all necessary libraries (e.g., streamlit, transformers, torch, matplotlib).
2. License: If you haven't already, create a LICENSE file in your repository and specify the MIT License or any other license you prefer.
3. Badges: You can add more badges related to your specific use case, such as build status or code quality.
4. Feel free to modify any sections to better fit your project's specifics or personal style!
5. Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
6. This README includes detailed instructions on installation, usage, contributing, and licensing, along with badges similar to your previous project.





