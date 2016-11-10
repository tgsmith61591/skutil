from jinja2 import Template, Environment, FileSystemLoader
from sklearn.externals import six
from sklearn.base import BaseEstimator
from skutil.base import overrides
from abc import ABCMeta, abstractmethod
import os

__all__ = [
    'EnvironmentTemplater',
    'StringTemplater'
]


class _BaseTemplater(six.with_metaclass(ABCMeta, BaseEstimator)):
    """The class for building a template from a string or a file given a dictionary
     of substitutions.

        * The ``build_template_from_string`` method returns the rendered template from a template string and
          a dictionary of substitutions.

        * The ``build_template_from_env`` method returns the rendered template from a template file and
          a dictionary of substitutions. The ``jinja2.Environment`` is created by specifying a template
          directory for the ``FileSystemLoader``.
    """
    @abstractmethod
    def build(self, string, params):
        raise NotImplementedError('This must be implemented by a subclass!')


class EnvironmentTemplater(_BaseTemplater):
    @overrides(_BaseTemplater)
    def build(self, template_file, params):
        """Generates the rendered template from a template 
        file and a dictionary of substitutions.

        Parameters
        ----------

        template_file: file
            The file located within ``skutil.h2o.templates``

        params: dict
            The dictionary of substitutions, where each key is 
            substituted with its corresponding value.

        Returns
        -------
        
        template : ``Template``
            Rendered Jinja template with substitutions from 
            ``dictionary`` made within ``template_string``.
        """

        templates_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.join(os.path.sep,"..","templates")
        env = Environment(autoescape=False, loader=FileSystemLoader(templates_dir))
        template = env.get_template(template_file)
        return template.render(params)


class StringTemplater(_BaseTemplater):
    @overrides(_BaseTemplater)
    def build(self, the_string, params):
        """Generates the rendered template from a template 
        string and a dictionary of substitutions.

        Parameters
        ----------

        the_string : str
            The string representation of the Jinja 
            template prior to substitutions.

        params : dict
            The dictionary of substitutions, where each key is 
            substituted with its corresponding value.

        Returns
        -------

        template : ``Template``
            Rendered Jinja template with substitutions from 
            ``dictionary`` made within ``template_string``.

        """
        template = Template(the_string)
        return template.render(params)
