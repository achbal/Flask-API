#!/usr/bin/env python3

"""
Create an API using Flask to calculate the estimation 
in auctions price for one or more vehicles

Script performs the following tasks:
- Create instance of web application
- Get request data 
- Turn the JSON output into a Response object
- Set the route to '/common/v1/bca-quotation/' and set HTTP methods to POST. 
- Show the estmation cost in JSON form as response.

@author  Oualid Achbal, Back-end Developer , Autobiz 2018
"""

# -------------------------------
#  Imports of standard modules --
# -------------------------------
import logging
import sys
import warnings
from os import path
import sys

# ----------------------------
# Imports for other modules --
# ----------------------------
from flask import Flask, jsonify, request
import subprocess
#sys.path.append(path.abspath('/FunctionsBCA'))
#from FunctionsBCA import *
#from FunctionsBCA import model_BCA_API

# Create an instance of our web application
app = Flask(__name__)

# endpoint to create new field
@app.route("/common/v1/bca-quotation/", methods=["POST"])
def add_field():
    """
    Add a field to the database and return the prediction value
    """
    km = request.json['km']
    makeId = request.json['makeId']
    modelId = request.json['modelId']
    fuelId = request.json['fuelId']
    bodyId = request.json['bodyId']
    door = request.json['door']
    liter = request.json['liter']
    quotation = request.json['quotation']
    rotation = request.json['rotation']
    refurbishmentCost = request.json['refurbishmentCost']
    powerDin = request.json['powerDin']
    year = request.json['year']
    month = request.json['month']
    sellDate = request.json['sellDate']


    # Command to call into subprocess
    space = " "
    sequence = (km, makeId, modelId,fuelId, bodyId, door, liter, quotation, rotation, refurbishmentCost, powerDin, year, month, sellDate)
    cmd = "python3 model_BCA_API.py " + space.join(sequence)
    logging.debug("The command is: {}".format(cmd))
    try: 
        result = subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE).communicate()[0]
        result = result.decode("utf-8").rstrip() 
        logging.info("The BCA quatation is: {}".format(result)) 
    except subprocess.CalledProcessError as e: 
        return "An error occurred." 

    return jsonify(bcaQuotation = result)


if __name__ == '__main__':
    try:
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(name)-15s %(message)s', level=logging.INFO)
        # Disable request package logger
        logging.getLogger("requests").setLevel(logging.ERROR)
        # Disable warnings
        warnings.filterwarnings("ignore")
        # Run app
        app.run(debug=True,host='0.0.0.0',port=8080)
    except Exception as exc:
        logging.critical('Exception occured: %s', exc, exc_info=True)
        sys.exit(1)

