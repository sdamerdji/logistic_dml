# logistic_dml
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]

Python implementation of "Double/Debiased Machine Learning for Logistic Partially Linear Model" by Molei Liu, Yi Zhang and Doudou Zhou. See https://academic.oup.com/ectj/article/24/3/559/6296639

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Documentation and Maintenance

**<font size="3">logistic_dml</font>** is currently maintained by <a href="#https://github.com/sdamerdji">@sdamerdji</a>.
<p>Bugs can be reported to the issue tracker at <a href="https://github.com/sdamerdji/logistic_dml/issues">https://github.com/sdamerdji/logistic_dml/issues</a> </p>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation
**<font size="3">logistic_dml</font>**  requires:
* Python (>=3.9)
* Pandas (>=1.5.2)
* SciPy (>=1.9.3)
* NumPy (>=1.23.5)
* scikit-learn (>=1.3.0)

To install **logistic_dml** with pip use
   ```sh
   pip install logistic_dml
   ```
**logistic_dml** can be installed from source via

   ```sh
   git clone https://github.com/sdamerdji/logistic_dml.git
   cd logistic_dml
   pip install .
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
Example:
   ```sh
        from logistic_dml import DML  
        Y = np.array([1, 1, 1, 1, 0, 0, 1, 0, 1, 0]*2)
        A = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1]*2)
        X = pd.DataFrame({'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*2,
                          'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]*2})
        K = 2
        model1 = LogisticRegression()
        model2 = LinearRegression()
        result = DML(classifier=model1, regressor=model2,random_seed=0).dml(Y, A, X, k_folds=K)      
  ```
    
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

[insert text here]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Testing

After installation, you can launch the test suite from inside the source directory,
   ```sh
    python test_logistic_dml.py
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License
[insert text here]
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Email: salim.damerdji@stats.ox.ac.uk

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/sdamerdji/logistic_dml/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]:https://github.com/sdamerdji/logistic_dml/forks
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/sdamerdji/logistic_dml/issues



