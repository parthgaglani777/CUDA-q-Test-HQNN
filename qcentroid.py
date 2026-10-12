#
# QCentroid entrypoint
# 
# DO NOT change the name of this file.
# DO NOT edit the signature of the run() function.
#
#
# TODO: Import here any non-standard packages you need or your custom modules
#


# Configure the QCentroid logging feature to see log traces in the dashboard
import logging
logger = logging.getLogger("qcentroid-user-log")


def run(input_data:dict, solver_params:dict, extra_arguments:dict) -> dict:

    logger.info("Start of my solver...")

    #
    # Add your solver's code here, or call it from here if it is already implemented in another module
    #

    output = {"message": "Hello world!"}

    logger.info("End of my solver, returning output.")

    # And this is the output it returns. This output must be a dictionary.
    return output
