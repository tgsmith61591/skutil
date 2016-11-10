from __future__ import print_function, absolute_import, division
from skutil.h2o.java_templating import StringTemplater, EnvironmentTemplater
import os
import sys


def test_templater():
    templates_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.join(os.path.sep, "..","..","templates")
    template_filename = "main_class.java"

    # attempt to load template file
    try:
        with open(os.path.join(templates_dir, template_filename), 'r') as myfile:
            template_string = myfile.read()
    except IOError as io_err:
        print("\n\n" + io_err.strerror + ": " + template_filename + "\n")
        sys.exit(-1)

    # replace values within template with values in kwargs
    kwargs = {"model_class_name": "gbm_test",
              "fields": """
        RowData row = new RowData();
        row.put("Year", "1987");
        row.put("Month", "10");
        row.put("DayofMonth", "14");
        row.put("DayOfWeek", "3");
        row.put("CRSDepTime", "730");
        row.put("UniqueCarrier", "PS");
        row.put("Origin", "SAN");
        row.put("Dest", "SFO");
        """}

    # render same template file using the string representation of the template
    from_string = StringTemplater().build(template_string, params=kwargs)

    # render same template file using the jinja2.Environment object
    from_env = EnvironmentTemplater().build(template_filename, params=kwargs)

    # they should be the same
    assert(from_string == from_env) # both templates should be equivalent
