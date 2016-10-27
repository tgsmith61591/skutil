from jinja2 import Template, Environment, FileSystemLoader
from sklearn.externals import six
from abc import ABCMeta
import os


class Templater(six.with_metaclass(ABCMeta, object)):
    """The class for building a template from a string or a file given a dictionary
     of substitutions.

        * The ``build_template_from_string`` method returns the rendered template from a template string and
          a dictionary of substitutions.

        * The ``build_template_from_env`` method returns the rendered template from a template file and
          a dictionary of substitutions. The ``jinja2.Environment`` is created by specifying a template
          directory for the ``FileSystemLoader``.
    """

    @staticmethod
    def build_template_from_string(template_string, dictionary):
        """Generates the rendered template from a template string and a dictionary of substitutions.

        Parameters
        ----------

        template_string : str
            The string representation of the Jinja template prior to substitutions.

        dictionary : dict
            The dictionary of substitutions, where each key is substituted with its corresponding value.

        Returns
        -------

        Rendered Jinja template with substitutions from ``dictionary`` made within ``template_string``.

        """

        template = Template(template_string)
        return template.render(dictionary)

    @staticmethod
    def build_template_from_env(template_file, dictionary):
        """Generates the rendered template from a template file and a dictionary of substitutions.

        Parameters
        ----------

        template_file: file
            The file located within skutil.h2o.templates

        dictionary: dict
            The dictionary of substitutions, where each key is substituted with its corresponding value.

        Returns
        -------

        Rendered Jinja template with substitutions from ``dictionary`` made within ``template_string``.
        """

        templates_dir = os.path.dirname(os.path.realpath(__file__)) + "/../templates"
        env = Environment(autoescape=False, loader=FileSystemLoader(templates_dir))
        template = env.get_template(template_file)
        return template.render(dictionary)
