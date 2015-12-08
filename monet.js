
var app = {};

(function(){
    d3.csv("monet_data_w_cluster.csv", function(error, data){
    	if (error) throw error;

    	// All 5 colours are saved as part of a string
    	// convert to array (length 5) of arrays (length 3, RGB) 
    	data.map(function(x){
    		re = /\d+,\s*\d+,\s*\d+/g;
	    	x.cols = x.cluster_count.match(re);
	    	if (x.cols){
		    	x.cols = x.cols.map(function(str){
		    		return str.split(',').map(function(s){ return parseInt(s)} )
		    	})
    		}
    	})

    	data = data.filter(function(d) { 
    		return (!(d.cols===null)) })

    	// Some dates were given as a range.
    	// Select first year given, then convert to integer
    	data.map(function(x){

    		re = /\d{4}/;
    		x.yr = parseInt( x.year.match(re) );
    		if(x.yr % 2 === 1){
    			x.yr += -1
    		}

    	})

    	// Filter out NaN years
    	data = data.filter(function(d){ return (!isNaN(d.yr) && d.yr>1859 && d.yr<1926) })

    	// Year will translate into the x-axis coordinate.
    	// The 'y-axis' be defined by stacking the image icons vertically
    	//   so I need an index for the image within the year.
    	//   Since I have no information about what order these images were
    	//   painted in over the course of the given year, I will simply 
    	//   use a count of the position within the data array.

		data = _.sample(data, 150);

    	var count_years = {};
    	data.forEach(function(image, index){
    		var key = image.yr.toString();
    		var i = count_years.hasOwnProperty(key) ? count_years[key] : 0;
    		count_years[key] = i + 1;
    		image.yri = i;
    	})
    	// Note that a few of the dates are missing (.yr = NaN) and one is non-sensical (1980)

    	// Image Size
    	var margin = {top: 20, right: 30, bottom: 60, left: 30},
	    width = 1240 - margin.left - margin.right,
	    height = 680 - margin.top - margin.bottom;
		var radius = 18,
		    padding = 1;

	    // x-axis and y-axis begin scalse
	    var x = d3.scale.linear()
	        .range([0, width]);

	    var y = d3.scale.linear()
	        .range([height, 0]);

	    var col = ["#98abc5", "#8a89a6", "#7b6888", "#6b486b", "#a05d56", "#d0743c", "#ff8c00"]

		// Compute the scales domains. choose min & max year / year index accordingly
		x.domain([1860, 1925]);
		y.domain(d3.extent(data, function(d) { return d.yri; })).nice();

		var arc = d3.svg.arc()
		    .outerRadius(radius)
		    .innerRadius(radius - 13);

		var pie = d3.layout.pie()
		    .sort(null)
		    .value(function(x){
		    	return x.data;
		    });

		app.data = data;


		function rotTween(d,i,a) {
		    var inter_fun = d3.interpolate(0, 360);
		    return function(t) {
		    	return "translate(" + (x(d.yr)) + "," + (height - d.yri*2*(radius + padding)) + ") " + 
		         					"rotate(" + inter_fun(t) + ",0,0)";
		    };
		}

		function bindSpinner(pie){

			var intervalID;

			pie.on('mouseover.bindspin', function(){
			    		var that = this;
			    		d3.select(that)
			    			.transition()
        					.duration(900)
        					.ease('linear')
			    			.attrTween("transform", rotTween)
			    		intervalID = setInterval(function(){
			    			d3.select(that)
			    			.transition()
        					.duration(900)
        					.ease('linear')
			    			.attrTween("transform", rotTween)
			    		},900);

					})
			   	.on('mouseout.bindspin', function(){
			   		clearInterval(intervalID);
			   	})

		}

		function previewImage(pie){
			pie.on('mouseover.preview', function(){
					var file = d3.select(this).datum().file_id_full;
					var first = d3.select(this).datum().title + " (" + d3.select(this).datum().year + ")";
					var second = "";
					if (first.length > 50){
						re = /\s+/g;
						var cap_split = first.split(re);
						first = cap_split.slice(0, Math.ceil(cap_split.length / 2)).join(" ")
						second = cap_split.slice(Math.ceil(cap_split.length / 2), cap_split.length).join(" ")
					}

		    		d3.select('#imageinfo img')
			    			.attr("height", 0)
		    				.attr('src', function(){	    					
		    					var imagepath = 'images/'+file+'.jpg'
		    					return imagepath
		    				})
			    			.attr("class", "view-image")
			    			.transition()
	        					.duration(800)
			    			.attr("height", 260)

					d3.select('.caption.line1')
							.text(first)
							.transition()
								.duration(800)
							.styleTween("font-size", function() {
								  return d3.interpolate(
								    this.style.getPropertyValue("font-size"),
								    "20px"
								  );
								})
					d3.select('.caption.line2')
						.text(second)
						.transition()
							.duration(800)
						.styleTween("font-size", function() {
							  return d3.interpolate(
							    this.style.getPropertyValue("font-size"),
							    "20px"
							  );
							})
					}
				).on('mouseout.preview', function(){ 

			   		d3.select('#imageinfo img')
		    			.transition()
        					.duration(400)
		    			.attr("height", 0)

	    			d3.selectAll('.caption')
	    					.transition()
	    						.duration(400)
	    					.styleTween("font-size", function() {
	    						  return d3.interpolate(
	    						    this.style.getPropertyValue("font-size"),
	    						    "0px"
	    						  );
	    						})

	    			// d3.select('#imageinfo #caption')
	    			// 		.text("")
	    			// 		.attr('class', '#showcaption')

			   	})
		}

		// Initialize chart area
		var svg = d3.select("#chart").append("svg")
			    .attr("width", width + margin.left + margin.right)
			    .attr("height", height + margin.top + margin.bottom)
			.append("g")
		    	.attr("transform", "translate(" + margin.left + "," + margin.top + ")")
			.selectAll(".pie")
            	.data(data)
	        .enter().append("g")
			    .attr("transform", function(d){
			        return "translate(" + x(d.yr) + "," + (height - d.yri*2*(radius + padding)) + ")"})
			    .attr("class", "pie")
			    .attr("data-imgfull", function(d){
			    	return d.file_id_full
			    })
			    .attr("data-imgth", function(d){
			    	return d.file_id_thumb
			    })
			    .attr("width", radius * 2)
			    .attr("height", radius * 2)
			    .call(previewImage)
			    .call(bindSpinner)
		    .append("g")
		    	.attr("class", "piebits")
		    	.attr("transform", "translate(" + 0 + "," + 0 + ")")
			.selectAll(".arc")
	            .data(function(d, i) { 
		              return pie(d.cols.map( function(x){
				              	return {
				              		data: 1,
				              		color: "rgb(" + x[0] + "," + x[1] + "," + x[2] + ")"
				              	};
				              }) 
		              	);
		             })
	          	.enter().append("path")
		            .attr("class", "arc")
		            .attr("d", arc)
		            .style("fill", function(d,i) {
		            	return d.data.color });

		d3.selectAll(".piebits")
			.append("rect")
				.attr("x", -radius)
				.attr("y", -radius)
				.attr("width", 2*radius)
				.attr("height", 2*radius)
				.attr("fill-opacity", 0)

		var xAxis = d3.svg.axis().scale(x).orient("bottom").tickFormat(d3.format("d"));

	    // // Add the x-axis.
	    d3.select("#chart svg")
	    	.append("g")
	        .attr("class", "x axis")
	        .attr("transform", "translate(" + (margin.left) + "," + (height + margin.bottom - 8) + ")")
	        .call( xAxis );

	    app.data = data;

    });
})()