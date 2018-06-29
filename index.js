

var express = require("express"),
    bodyParser = require("body-parser"),
    request = require("request"),
    cheerio = require("cheerio"),
    exec = require("child_process").exec;

var app=express();
app.set("view engine", "ejs");
app.use(express.static("public"));
app.use(bodyParser.urlencoded({extended: true}));

app.get("/", function(req, res){
  res.render("home",{movies:[]});
});

app.post("/", function(req, res){
  movies=[];
  console.log("[+] Movie search: "+req.body.q);
  request("http://www.omdbapi.com/?apikey=56077564&s="+req.body.q,function(error,response,body){
    if(!error && response.statusCode==200){
      results = JSON.parse(body);
      if(results.Response == "False")
        res.redirect("notFound");
      else{
        movies=results.Search;
        res.render("home",{movies:movies});
      }
    }
  });
});

app.get("/:id/show", function(req, res){
  console.log("Route Called!");
  var startTime = new Date();
  request("http://www.imdb.com/title/"+req.params.id+"/reviews/_ajax", function(err, response, html){
    var reviews=[];

    if(err)
      console.log(err);

    // Load cherio to scrape the html dom
    var $ = cheerio.load(html);
    var counter=0;
    var len = $(".text").length;
    $(".text").each(function(i, elem){
      var review=$(this).text();
      review = review.replace(/[.,\/#!$%\^&\*;:{}=\-_`'"~()]/g,"")
      exec('python "./externalScripts/predict.py" "'+review+'"', function(err, stdout, stderr){
        var ans=String(stdout);
        reviews.push({review: review, rating: ans});
        counter++;
        if(counter==len){
          var endTime=new Date();
          console.log("All done, Seconds: ",(endTime.getTime()-startTime.getTime())/1000);
          res.send(reviews);
        }
      });
    });
  });
});

app.listen(80, function(){
  console.log("Server is listenting");
  
});