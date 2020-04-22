# Kaggle Dogs vs. Cats Competition

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=FFFFFF)](https://www.linkedin.com/in/costaruan/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=FFFFFF)](https://www.kaggle.com/costaruan/)

[![Repo size](https://img.shields.io/github/repo-size/costaruan/kaggle-dogs-vs-cats-competition)](https://github.com/costaruan/kaggle-dogs-vs-cats-competition/)
[![Languages](https://img.shields.io/github/languages/count/costaruan/kaggle-dogs-vs-cats-competition)](https://github.com/costaruan/kaggle-dogs-vs-cats-competition/)
[![License](https://img.shields.io/github/license/costaruan/kaggle-dogs-vs-cats-competition)](https://github.com/costaruan/kaggle-dogs-vs-cats-competition/blob/master/LICENSE.md)
[![Made by costaruan](https://img.shields.io/badge/made%20by-costaruan-green)](https://github.com/costaruan/kaggle-dogs-vs-cats-competition/)

In this competition, you'll write an algorithm to classify whether images contain either a dog or a cat. This is easy for humans, dogs, and cats. Your computer will find it a bit more difficult.

[Here](https://www.kaggle.com/c/dogs-vs-cats) you'll find all information and data to understand the challenge.

## The Asirra Dataset

Web services are often protected with a challenge that's supposed to be easy for people to solve, but difficult for computers. Such a challenge is often called a [CAPTCHA](http://www.captcha.net/) (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof). HIPs are used for many purposes, such as to reduce email and blog spam and prevent brute-force attacks on web site passwords.

Asirra (Animal Species Image Recognition for Restricting Access) is a HIP that works by asking users to identify photographs of cats and dogs. This task is difficult for computers, but studies have shown that people can accomplish it quickly and accurately. Many even think it's fun! Here is an example of the Asirra interface:

Asirra is unique because of its partnership with [Petfinder.com](http://www.petfinder.com/), the world's largest site devoted to finding homes for homeless pets. They've provided Microsoft Research with over three million images of cats and dogs, manually classified by people at thousands of animal shelters across the United States. Kaggle is fortunate to offer a subset of this data for fun and research.

## Image Recognition Attacks

While random guessing is the easiest form of attack, various forms of image recognition can allow an attacker to make guesses that are better than random. There is enormous diversity in the photo database (a wide variety of backgrounds, angles, poses, lighting, etc.), making accurate automatic classification difficult. In an informal poll conducted many years ago, computer vision experts posited that a classifier with better than 60% accuracy would be difficult without a major advance in the state of the art. For reference, a 60% classifier improves the guessing probability of a 12-image HIP from 1/4096 to 1/459.

## State of the Art

The current literature suggests machine classifiers can score above 80% accuracy on this task [[1]](http://xenon.stanford.edu/~pgolle/papers/dogcat.pdf). Therfore, Asirra is no longer considered safe from attack. We have created this contest to benchmark the latest computer vision and deep learning approaches to this problem.

---

**Created by** [`costaruan`](https://costaruan.dev/)
