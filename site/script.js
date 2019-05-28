

// for zooming
function imageZoom(imgID, imgID_2, imgID_3, resultID, resultID_2, resultID_3) {
  let img, lens, result, cx, cy;
  img = document.getElementById(imgID);
  result = document.getElementById(resultID);
  /* Create lens: */
  lens = document.createElement("DIV");
  lens.setAttribute("class", "img-zoom-lens");
  /* Insert lens: */
  if(index_image == 0)
    img.parentElement.insertBefore(lens, img);
  /* Calculate the ratio between result DIV and lens: */
  cx = result.offsetWidth / lens.offsetWidth;
  cy = result.offsetHeight / lens.offsetHeight;
  /* Set background properties for the result DIV */
  result.style.backgroundImage = "url('" + img.src + "')";
  result.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";
  /* Execute a function when someone moves the cursor over the image, or the lens: */
  

  
  // for the second
  let img_2, lens_2, result_2, cx_2, cy_2;
  img_2 = document.getElementById(imgID_2);
  result_2 = document.getElementById(resultID_2);
  /* Create lens: */
  lens_2 = document.createElement("DIV");
  lens_2.setAttribute("class", "img-zoom-lens");
  /* Insert lens: */
  if(index_image == 0)
    img_2.parentElement.insertBefore(lens_2, img_2);
  /* Calculate the ratio between result DIV and lens: */  
  cx_2 = result_2.offsetWidth / lens_2.offsetWidth;
  cy_2 = result_2.offsetHeight / lens_2.offsetHeight;
  /* Set background properties for the result DIV */
  result_2.style.backgroundImage = "url('" + img_2.src + "')";
  result_2.style.backgroundSize = (img_2.width * cx_2) + "px " + (img_2.height * cy_2) + "px"; 
  
  // fot the third one 
  let img_3, lens_3, result_3, cx_3, cy_3;
  img_3 = document.getElementById(imgID_3);
  result_3 = document.getElementById(resultID_3);
  /* Create lens: */
  lens_3 = document.createElement("DIV");
  lens_3.setAttribute("class", "img-zoom-lens");
  /* Insert lens: */
  if(index_image == 0)
    img_3.parentElement.insertBefore(lens_3, img_3);
  /* Calculate the ratio between result DIV and lens: */  
  cx_3 = result_3.offsetWidth / lens_3.offsetWidth;
  cy_3 = result_3.offsetHeight / lens_3.offsetHeight;
  /* Set background properties for the result DIV */
  result_3.style.backgroundImage = "url('" + img_3.src + "')";
  result_3.style.backgroundSize = (img_3.width * cx) + "px " + (img_3.height * cy) + "px";
  
  
  
  lens.addEventListener("mousemove", function(e) { 
  moveLens(e, img, img, lens, result, cx, cy); 
  moveLens(e, img_2, img, lens_2, result_2, cx_2, cy_2);  
  moveLens(e, img_3, img, lens_3, result_3, cx_3, cy_3);}
  );
  
  img.addEventListener("mousemove", function(e) { 
  moveLens(e, img, img, lens, result, cx, cy); 
  moveLens(e, img_2, img, lens_2, result_2, cx_2, cy_2);  
  moveLens(e, img_3, img, lens_3, result_3, cx_3, cy_3);}
  );
  
 
  lens_2.addEventListener("mousemove", function(e)  { 
  moveLens(e, img, img_2, lens, result, cx, cy); 
  moveLens(e, img_2, img_2, lens_2, result_2, cx_2, cy_2);  
  moveLens(e, img_3, img_2, lens_3, result_3, cx_3, cy_3);
  }); 
  
  img_2.addEventListener("mousemove", function(e) { 
  moveLens(e, img, img_2, lens, result, cx, cy); 
  moveLens(e, img_2, img_2, lens_2, result_2, cx_2, cy_2);  
  moveLens(e, img_3, img_2, lens_3, result_3, cx_3, cy_3);
  }); 
  
  lens_3.addEventListener("mousemove", function(e)  { 
  moveLens(e, img, img_3, lens, result, cx, cy); 
  moveLens(e, img_2, img_3, lens_2, result_2, cx_2, cy_2);  
  moveLens(e, img_3, img_3, lens_3, result_3, cx_3, cy_3);
  }); 
  
  img_3.addEventListener("mousemove", function(e) { 
  moveLens(e, img, img_3, lens, result, cx, cy); 
  moveLens(e, img_2, img_3, lens_2, result_2, cx_2, cy_2);  
  moveLens(e, img_3, img_3, lens_3, result_3, cx_3, cy_3);
  }); 
  
 
  function moveLens(e, img, cusor_img, lens, result, cx, cy) {
    let pos, x, y;
    /* Prevent any other actions that may occur when moving over the image */
    e.preventDefault();
    /* Get the cursor's x and y positions: */
    pos = getCursorPos(cusor_img, e);
    /* Calculate the position of the lens: */
    x = pos.x - (lens.offsetWidth / 2);
    y = pos.y - (lens.offsetHeight / 2);
    /* Prevent the lens from being positioned outside the image: */
    if (x > img.width - lens.offsetWidth) {x = img.width - lens.offsetWidth;}
    if (x < 0) {x = 0;}
    if (y > img.height - lens.offsetHeight) {y = img.height - lens.offsetHeight;}
    if (y < 0) {y = 0;}
    /* Set the position of the lens: */
    lens.style.left = x + "px";
    lens.style.top = y + "px";
    /* Display what the lens "sees": */
    result.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px"; 
  }
  function getCursorPos(img, e) {
    let a, x = 0, y = 0;
    e = e || window.event;
    /* Get the x and y positions of the image: */
    a = img.getBoundingClientRect();
    /* Calculate the cursor's x and y coordinates, relative to the image: */
    x = e.pageX - a.left;
    y = e.pageY - a.top;
    /* Consider any page scrolling: */
    x = x - window.pageXOffset;
    y = y - window.pageYOffset;
    return {x : x, y : y};
  }
}
  
// end zooming 

var original_path = "images/original";
var our_path = "images/our";
var lanczos_path = "images/lanczos";
var index_image = -1;
var imgs = ['0.png', '1.png'] 
var currentdate = new Date(); 
 

in_progress = false;
window.onbeforeunload = function() {
if (in_progress) {
    return "You did not finish adnotating. Are you sure you want to leave?";
} else {
    return;
}
};
 
function sendAnswer(resp) {
    let date = "h2z-"+ currentdate.getDate() + (currentdate.getMonth()+1) + currentdate.getFullYear() + '-' + currentdate.getHours() +  currentdate.getMinutes() + currentdate.getSeconds() + Math.random().toString(36).substr(2, 5);;
    var result = jQuery.ajax({
        type: "POST",
        url: "http://localhost:8000",
        data: {'img': imgs[index_image], 'method': resp, 'file': date }
        });
    return result;
}

function answer(method) { 
    $('.btn').prop('disabled', true); 
    var ans = sendAnswer(method);   
    ans.done(function(msg) {
        get_the_next_image();
        $('.btn').prop('disabled', false);
    });
    ans.fail(function(err) {
        alert( "error", err );
    }); 
}
 
function select(method){  
    alert(method) 
    
    // i show send a post to a link with method and  index_image
    get_the_next_image();
}     

function end() {
    in_progress = false;
    $('#content').hide();
    $('.end').show();
}

function start() {
    in_progress = true;
    get_the_next_image(); 
    $('#startbtn').hide();
    $('#content').show();
}

function get_the_next_image(){ 
    index_image = index_image + 1;
    
    if(imgs[index_image]){  
        document.getElementById("original").src = original_path + "/" + imgs[index_image];  
        
        $("#left_btn").replaceWith($("#left_btn").clone())
        $("#right_btn").replaceWith($("#right_btn").clone())
         
        if (Math.random() < 0.5) { 
            left = document.getElementById("left") 
            left.src = our_path + "/" + imgs[index_image];
            document.getElementById("left_btn").addEventListener("click",  function() {  answer('our');}) 
            
            right = document.getElementById("right")
            right.src = lanczos_path + "/" + imgs[index_image];
            document.getElementById("right_btn").addEventListener("click", function() {   answer('lanczos');})  
             
        }
        else {
            left = document.getElementById("left")
            left.src = lanczos_path + "/" + imgs[index_image];
            document.getElementById("left_btn").addEventListener("click", function() {  
            answer('lanczos');}) 
            
            right = document.getElementById("right")
            right.src = our_path + "/" + imgs[index_image];
            document.getElementById("right_btn").addEventListener("click", function() { 
            answer('our');})  
        }
    }
    else {
          end();
    }
     
    imageZoom("original", "left", "right", "original_zoom", "left_zoom", "right_zoom");  
    
    $('#counter').text("Image " + (index_image + 1) + " of " + imgs.length);
}