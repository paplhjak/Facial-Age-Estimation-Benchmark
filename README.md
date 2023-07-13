<a name="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/paplhjak/Facial-Age-Estimation-Benchmark">
    <img src="doc/logo.png" alt="Logo" width="300" height="300">
  </a>

<h1 align="center">Facial Age Estimation Benchmark</h1>
  <p align="center">
    <a href="https://github.com/paplhjak/Facial-Age-Estimation-Benchmark/issues">Report Bug</a>
    ·
    <a href="https://github.com/paplhjak/Facial-Age-Estimation-Benchmark/issues">Request Feature</a>    
  </p>
</div>

<div>
  <p align="center">
  arXiv: <a href="https://arxiv.org/abs/2307.04570">Unraveling the Age Estimation Puzzle: Comparative Analysis of Deep Learning Approaches for Facial Age Estimation</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository serves as the official PyTorch codebase for the paper titled "Unraveling the Age Estimation Puzzle: Comparative Analysis of Deep Learning Approaches for Facial Age Estimation". You can find the paper on [arXiv](https://arxiv.org/abs/2307.04570) here. 

Comparing different age estimation methods poses a challenge due to the unreliability of published results, stemming from inconsistencies in the benchmarking process. Previous studies have reported continuous performance improvements over the past decade using specialized methods; however, our findings challenge these claims.

<center>
<img src="doc/MAE_vs_year.png" alt="Graph" height="300">  
</center>
    
We argue that, for age estimation tasks outside of the low-data regime, designing specialized methods is unnecessary, and the standard approach of utilizing cross-entropy loss is sufficient. 

<!-- GETTING STARTED -->
## Getting Started

This README is designed to cater to three types of users:
- Users who only want to use the same data splits to ensure comparability with previous state-of-the-art methods, see ["Using the Data Splits"](doc/using_the_data_splits.md)
- Users who want to train the implemented age estimation models using this repository, see ["Using the Repository"](doc/using_the_repository.md)
- Users who want to implement their own methods, use this repository, and compare their results against the state-of-the-art, see ["Implementing New Methods"](doc/implementing_new_methods.md)

For each user type, a dedicated walk-through is provided. Click on the links above to access the relevant sections.

<!-- CONTRIBUTING -->
## Contributing
Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Alternatively, contact us over email :cowboy_hat_face:

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->    
## Contact

For questions/comments please email Jakub Paplham  at paplhjak@fel.cvut.cz :slightly_smiling_face:

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[PyTorch.js]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/

