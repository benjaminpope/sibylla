# sibylla
Gradient Descent Image Reconstruction

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope)

## What is sibylla?

sibylla - currently a placeholder - is going to be a one-stop-shop for Jax astronomical image restoration code, built on [zodiax](https://louisdesdoigts.github.io/zodiax/) to be object-oriented and play nicely with both pixel-level optical models in [dLux](https://louisdesdoigts.github.io/dLux/) and (yet to be open-sourced) VLBI modelling.

## Installation

sibylla is hosted on PyPI (though this is currently a placeholder): the easiest way to install this is with 

```
pip install sibylla
```

You can also build from source. To do so, clone the git repo, enter the directory, and run

```
pip install .
```

We encourage the creation of a virtual enironment to run sibylla to prevent software conflicts as we keep the software up to date with the lastest version of the core packages.


## Use & Documentation

Documentation will be found [here](https://benjaminpope.github.io/sibylla/), though this is currently a placeholder. 

## Collaboration & Development

We are always looking to collaborate and further develop this software! We have focused on flexibility and ease of development, so if you have a project you want to use sibylla for, but it currently does not have the required capabilities, don't hesitate to [email me](b.pope@uq.edu.au) and we can discuss how to implement and merge it! Similarly you can take a look at the `CONTRIBUTING.md` file.

## Name

Why is it called sibylla?

Sibylla - Latin for the Sibyl - was Aeneas' guide in his *descent* into the underworld. She uttered the famous line *facilis descensus Averno* - the descent to Hell is easy (but coming back is hard!), and warned him about the gate of false visions. It is our goal to make gradient descent image reconstruction easy, and to probabilistically quantify uncertainties and avoid those false visions!
