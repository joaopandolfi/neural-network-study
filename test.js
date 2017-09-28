var convnetjs = require("convnetjs");
// species a 2-layer neural network with one hidden layer of 20 neurons
var layer_defs = [];
// input layer declares size of input. here: 2-D data
// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
// then the first two dimensions (sx, sy) will always be kept at size 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'}); 
// declare the linear classifier on top of the previous hidden layer
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// forward a random data point through the network
var x = new convnetjs.Vol([0.3, -0.5,0.1]);
var prob = net.forward(x); 

// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101

var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
//trainer.train(x, 0); // train the network, specifying that x is class zero

for(i = 0 ; i<10; i++){
	trainer.train(x, 0); // train the network, specifying that x is class zero
	var prob2 = net.forward(x);
	console.log('probability that x is class 0: ' + prob2.w[0]);

}
// now prints 0.50374, slightly higher than previous 0.50101: the networks
// weights have been adjusted by the Trainer to give a higher probability to
// the class we trained the network with (zero)

//Checking Crescent or Decrescent List


//TESTS

var database = {data:[],class:[]};

//Crescent
database.data.push([0.1,0.2,0.3])
database.class.push(1)
database.data.push([0.2,0.3,0.4])
database.class.push(1)
database.data.push([0.11,0.22,0.33])
database.class.push(1)
database.data.push([0.12,0.23,0.34])
database.class.push(1)
database.data.push([0.01,0.02,0.03])
database.class.push(1)
database.data.push([0.8,0.881,0.889])
database.class.push(1)
database.data.push([0.7,0.8,0.9])
database.class.push(1)


//Decrescent
database.data.push([0.3,0.2,0.1])
database.class.push(2)
database.data.push([0.33,0.22,0.11])
database.class.push(2)
database.data.push([0.5,0.3,0.1])
database.class.push(2)
database.data.push([0.9,0.2,0.1])
database.class.push(2)

console.log("Learning");
for(learn = 0 ; learn < 1000; learn++){
	//Traning
	for(i = 0 ; i<database.class.length; i++){
		x = new convnetjs.Vol(database.data[i]);
		trainer.train(x, database.class[i]); // train the network, specifying that x is class zero
		var prob2 = net.forward(x);
		//console.log('probability that '+ i +' is (Cres) class 1: ' + prob2.w[1]);
		//console.log('probability that '+ i +' is (Decres) class 2: ' + prob2.w[2]);

	}
}
//Testing

var test = {data:[],class:[]};
test.data.push([0.001,0.002,0.7])
test.data.push([0.5,0.2,0.001])
test.data.push([0.8,0.8,0.9])
test.data.push([0.5,0.49999,0.4999])
test.data.push([0.7,0.8,0.9])


console.log(test.data.length)
console.log("Executing")
for(i = 0 ; i<test.data.length; i++){
	x = new convnetjs.Vol(test.data[i]);
	var prob2 = net.forward(x);
	console.log('Dataset   ' + test.data[i])
	console.log('probability that '+ i +' is (Cres) class 1: ' + prob2.w[1]);
	console.log('probability that '+ i +' is (Decres) class 2: ' + prob2.w[2]);
	console.log("")
}
