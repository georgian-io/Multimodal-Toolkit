.. multimodal toolkit documentation master file, created by
   sphinx-quickstart on Wed Sep  2 12:06:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Multimodal Transformers Documentation
==============================================
A toolkit for incorporating multimodal data on top of text data for classification
and regression tasks. This toolkit is heavily based off of `HuggingFace Transformers <https://huggingface.co/transformers/>`_.
It adds a combining module that takes the outputs of the transformers in addition to
categorical and numerical features to produce rich multimodal features
for downstream classification/regression layers.


See its documentation for specific details regarding HuggingFace transformer models, configs, and tokenizers.

.. image:: ./model_image.png

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction
   notes/combine_methods
   notes/colab_example

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/model
   modules/data


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
