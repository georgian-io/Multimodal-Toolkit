multimodal.data
=======================

The data module includes two functions to help load your own datasets
into :class:`multimodal.data.tabular_torch_dataset.TorchTabularTextDataset`
which can be fed into a :class:`torch.utils.data.DataLoader`. The
:obj:`multimodal.data.tabular_torch_dataset.TorchTabularTextDataset`'s
:obj:`__getitem__` method's outputs can be directly fed to the
forward pass to a model in :obj:`multimodal.model.tabular_transformers`.

.. Note::
    You may still need to move the :obj:`__getitem__` method outputs to the right gpu device.

multimodal.data.load\_data
---------------------------------

.. automodule:: multimodal.data.load_data
   :members:
   :undoc-members:
   :show-inheritance:

multimodal.data.tabular\_torch\_dataset
----------------------------------------------

.. automodule:: multimodal.data.tabular_torch_dataset
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: multimodal.data
   :members:
   :undoc-members:
   :show-inheritance:
