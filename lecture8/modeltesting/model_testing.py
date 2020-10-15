
from surprise import Dataset
from surprise.dump import load as surprise_model_load
import numpy as np
import torch
from torch import nn
import pandas as pd
import random
from uuid import uuid4
from flask import (
	Flask,
	session,
	request,
	redirect,
	url_for,
	render_template_string
)
from planout.experiment import SimpleExperiment
from planout.ops.random import *


class ModelExperiment(SimpleExperiment):
		def setup(self):
				self.set_log_file('model_abtest.log')
		def assign(self, params, userid):
				params.use_pytorch = BernoulliTrial(p=0.5, unit=userid)
				if params.use_pytorch:
						params.model_type = 'pytorch'
				else:
						params.model_type = 'surprise'

class MF(nn.Module):
		itr = 0
		
		def __init__(self, n_user, n_item, k=18, c_vector=1.0, c_bias=1.0):
				super(MF, self).__init__()
				self.k = k
				self.n_user = n_user
				self.n_item = n_item
				self.c_bias = c_bias
				self.c_vector = c_vector
				
				self.user = nn.Embedding(n_user, k)
				self.item = nn.Embedding(n_item, k)
				
				# We've added new terms here:
				self.bias_user = nn.Embedding(n_user, 1)
				self.bias_item = nn.Embedding(n_item, 1)
				self.bias = nn.Parameter(torch.ones(1))
		
		def __call__(self, train_x):
				user_id = train_x[:, 0]
				item_id = train_x[:, 1]
				vector_user = self.user(user_id)
				vector_item = self.item(item_id)
				
				# Pull out biases
				bias_user = self.bias_user(user_id).squeeze()
				bias_item = self.bias_item(item_id).squeeze()
				biases = (self.bias + bias_user + bias_item)
				
				ui_interaction = torch.sum(vector_user * vector_item, dim=1)
				
				# Add bias prediction to the interaction prediction
				prediction = ui_interaction + biases
				return prediction
		
		def loss(self, prediction, target):
				loss_mse = F.mse_loss(prediction, target.squeeze())
				
				# Add new regularization to the biases
				prior_bias_user =  l2_regularize(self.bias_user.weight) * self.c_bias
				prior_bias_item = l2_regularize(self.bias_item.weight) * self.c_bias
				
				prior_user =  l2_regularize(self.user.weight) * self.c_vector
				prior_item = l2_regularize(self.item.weight) * self.c_vector
				total = loss_mse + prior_user + prior_item + prior_bias_user + prior_bias_item
				return total

def get_top_n_pytorch(model,trainset,urid_input,n=10):
		
		testset = trainset.build_anti_testset()
		preds = []
		for urid, irid, _ in testset:
				if urid==urid_input:
						preds.append((irid,float(model(torch.tensor([[int(trainset.to_inner_uid(urid)),
																													int(trainset.to_inner_iid(irid))]]
																											 ))
																		))
												)
		# Then sort the predictions and retrieve the n highest ones.
		preds.sort(key=lambda x: x[1], reverse=True)
		return preds[:n]


def get_top_n_surprise(model,trainset,urid_input,n=10):
		
		testset = trainset.build_anti_testset()
		preds = []
		for urid, irid, _ in testset:
				if urid==urid_input:
						preds.append((irid,float(model.predict(urid, irid).est)))
		# Then sort the predictions and retrieve the n highest ones.
		preds.sort(key=lambda x: x[1], reverse=True)
		return preds[:n]


#Data
df = pd.read_csv('./movies.dat',sep="::",header=None,engine='python')
df.columns = ['iid','name','genre']
df.set_index('iid',inplace=True)
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

#Parameters for the pytorch model
lr = 1e-2
k = 10 #latent dimension
c_bias = 1e-6
c_vector = 1e-6

model_pytorch = MF(trainset.n_users, trainset.n_items, k=k, c_bias=c_bias, c_vector=c_vector)
model_pytorch.load_state_dict(torch.load('./pytorch_model'))
model_pytorch.eval()
_, model_surprise = surprise_model_load('./surprise_model')





app = Flask(__name__)

app.config.update(dict(
	DEBUG=True,
	SECRET_KEY='MODEL_TESTING_BY_THEJA_TULABANDHULA',
))


@app.route('/',methods=["GET"])
def main():
		# if no userid is defined make one up
		if 'userid' not in session:
				session['userid'] = str(random.choice(trainset.all_users()))

		model_perf_exp = ModelExperiment(userid=session['userid'])
		model_type = model_perf_exp.get('model_type')
		resp = {}
		resp["success"] = False

		print(model_type,resp,session['userid'])

		try:
				if model_type=='pytorch':
						user_ratings = get_top_n_pytorch(model_pytorch,trainset,session['userid'],n=10) 
				elif model_type=='surprise':
						user_ratings = get_top_n_surprise(model_surprise,trainset,session['userid'],n=10) 

				print(user_ratings)
				resp["response"] = [df.loc[int(iid),'name'] for (iid, _) in user_ratings]
				resp["success"] = True

				return render_template_string("""
				<html>
					<head>
						<title>Recommendation Service</title>
					</head>
					<body>
						<h3>
							Recommendations for userid {{ userid }} based on {{ model_type }} are shown below: <br>
						</h3>

						<p>

						{% for movie_item in resp['response'] %}
						      <h5> {{movie_item}}</h5>
						{% endfor %}

						</p>

						<p>
							What will be your rating of this list (rate between 1-10 where 10 is the highest quality)?
						</p>
						<form action="/rate" method="GET">
							$<input type="text" length="10" name="rate"></input>
							<input type="submit"></input>
						</form>
					<br>
					<p><a href="/">Reload without resetting my user ID. I'll get the same recommendations when I come back.</a></p>
					<p><a href="/reset">Reset my user ID so I am a different user and will get re-randomized into a new treatment.</a></p>
					</body>
				</html>
			""", userid=session['userid'], model_type=model_type, resp=resp)
		except:
			return render_template_string("""
			<html>
				<head>
					<title>Recommendation Service</title>
				</head>
				<body>
					<h3>
						Recommendations for userid {{ userid }} based on {{ model_type }} are shown below. <br>
					</h3>
					<p>
					{{resp}}
					</p>

					<p>
						What will be your rating of this list (rate between 1-10 where 10 is the highest quality)?
					</p>
					<form action="/rate" method="GET">
						<input type="text" length="10" name="rate"></input>
						<input type="submit"></input>
					</form>
				<br>
				<p><a href="/">Reload without resetting my user ID. I'll get the same recommendations when I come back.</a></p>
				<p><a href="/reset">Reset my user ID so I am a different user and will get re-randomized into a new treatment.</a></p>
				</body>
			</html>
			""", userid=session['userid'], model_type=model_type, resp=resp)

@app.route('/reset')
def reset():
		session.clear()
		return redirect(url_for('main'))

@app.route('/rate')
def rate():
		rate_string = request.args.get('rate')
		try:
				rate_val = int(rate_string)
				assert rate_val > 0 and rate_val < 11

				model_perf_exp = ModelExperiment(userid=session['userid'])
				model_perf_exp.log_event('rate', {'rate_val': rate_val})

				return render_template_string("""
					<html>
						<head>
							<title>Thank you for the feedback!</title>
						</head>
						<body>
							<p>You rating is {{ rate_val }}. Hit the back button or click below to go back to recommendations!</p>
							<p><a href="/">Back</a></p>
						</body>
					</html>
					""", rate_val=rate_val)
		except:
				return render_template_string("""
					<html>
						<head>
							<title>Bad rating!</title>
						</head>
						<body>
							<p>You rating could not be parsed. That's probably not a number between 1 and 10, so we won't be accepting your rating.</p>
							<p><a href="/">Back</a></p>
						</body>
					</html>
					""")


# start the flask app, allow remote connections
app.run(host='0.0.0.0')