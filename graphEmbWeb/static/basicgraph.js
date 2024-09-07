<!DOCTYPE html>
<html>
<head>
    <title>Graph Visualization</title>
    <script src="https://d3js.org/d3.v5.min.js"></script>
</head>
<body>
    <div id="graph">Basic Graph</div>
    <svg id="svgBasic" width="640" height="480"></svg>
    <script>
        fetch('/api/graph/')
            .then(result => result.json())
            .then((output) => {
                console.log('Output: ', output);
                const graph = output;

                var svg = d3.select("#svgBasic");
                var width = svg.attr("width");
                var height = svg.attr("height");
                svg.append("rect").attr("height",10).attr("width",10).style("fill","red");
                var link = svg
                    .append("g")
                    .attr("class", "links")
                    .selectAll("line")
                    .data(graph.links)
                    .enter()
                    .append("line")
                    .attr("stroke-width", function(d) {
                      return 3;
                    });

                var node = svg
                    .append("g")
                    .attr("class", "nodes")
                    .selectAll("circle")
                    .data(graph.nodes)
                    .enter()
                    .append("circle")
                    .attr("r", 5)
                    .attr("fill", function(d) {
                      return "red";
                    })
                .call(
                  d3
                    .drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended)
                );

                var simulation = d3
                    .forceSimulation(graph.nodes)
                    .force(
                      "link",
                      d3
                        .forceLink()
                        .id(function(d) {
                          return d.id;
                        })
                        .links(graph.links)
                    )
                    .force("charge", d3.forceManyBody().strength())
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .on("tick", ticked);

                function ticked() {
                    link
                      .attr("x1", function(d) {
                        return d.source.x;
                      })
                      .attr("y1", function(d) {
                        return d.source.y;
                      })
                      .attr("x2", function(d) {
                        return d.target.x;
                      })
                      .attr("y2", function(d) {
                        return d.target.y;
                      })
                       .style("stroke", "black");

                    node
                      .attr("cx", function(d) {
                        return d.x;
                      })
                      .attr("cy", function(d) {
                        return d.y;
                      });
                    }

                function dragstarted(d) {
                if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
                }

                function dragged(d) {
                d.fx = d3.event.x;
                d.fy = d3.event.y;
                }

                function dragended(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
                }


            })
            .catch(err => console.error('Fetch Error:', err));




    </script>
</body>
</html>


            const color = d3.scaleOrdinal(d3.schemeCategory10);
<button class="button1">按钮1</button>
<button id="button2">按钮 2</button>
<button id="button3">按钮 3</button>


        // 绘制散点图
        svg.selectAll("circle")
            .data(embeddings)
            .enter().append("circle")
            .attr("cx", function(d) { return xScale(d.x); })
            .attr("cy", function(d) { return yScale(d.y); })
            .attr("r", 5)  // 点的半径
            .style("fill", "steelblue");
			
			
       var x = d3.scaleLinear()
            .domain([d3.min(embeddings, d => d.x), d3.max(embeddings, d => d.x)]).nice()
            .range([marginLeft, tsnewidth - marginRight])
            .unknown(marginLeft);
        var y = d3.scaleLinear()
            .domain([d3.min(embeddings, d => d.y), d3.max(embeddings, d => d.y)]).nice()
            .range([tsneheight - marginBottom, marginTop])
            .unknown(tsneheight - marginBottom);
			
	        // 如果需要，为每个点添加标签
        svg.selectAll("text")
            .data(embeddings)
            .enter().append("text")
            .attr("x", function(d) { return x(d.x); })
            .attr("y", function(d) { return y(d.y); })
            .text(function(d) { return d.id; })
            .attr("font-family", "sans-serif")
            .attr("font-size", "11px")
            .attr("fill", "black");
			
			
			
			                 "#516b91",
                 "#59c4e6",
                 "#edafda",
                 "#93b7e3",
                 "#a5e7f0",
                "#cbb0e3"
				
				
				
				
				
				                     '#787464',
                    '#cc7e63',
                    '#724e58',
        '#d87c7c'
		
		
		["#516b91","#59c4e6","#edafda","#93b7e3","#a5e7f0","#cbb0e3"]
		
		
		                '1':'#efa18d',
                '2':'#919e8b',
                '3':'#d7ab82',
                '4':'#6e7074',
                '5': '#61a0a8',
                '6': '#4b565b'
				
				
				                '1': "#c1232b",
                '2': "#27727b",
                '3':"#b5c334",
                '4':"#e87c25",
                '5': "#fe8463",
                '6': "#fcce10"
				
				
				#33384d
				
				
				
				        var coordsList = embeddings.map(item => {
            return {
                name: item.id,
                value: item.coord,  // 散点图的坐标值
                groupId: item.group,
                label: item.name,
                itemStyle: {
                    color: getColorForGroup(item.group)  // 为每个组分配颜色
                }
            };
        });
		
		
		
		<div id="people-switcher">
    <button id="person0" onclick="switchPerson(0)">Person 1</button>
    <button id="person1" onclick="switchPerson(1)">Person 2</button>
    <button id="person2" onclick="switchPerson(2)">Person 3</button>
    <button id="person3" onclick="switchPerson(3)">Person 4</button>
    <button id="person4" onclick="switchPerson(4)">Person 5</button>
</div>


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'graphdata',
        'USER': 'graphdatauser',
        'PASSWORD': '123456',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

