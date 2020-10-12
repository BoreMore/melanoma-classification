$('#prediction_form').submit(function(e) {
    e.preventDefault();
    $.ajax({
        method: 'POST',
        url: 'http://127.0.0.1:5000/api/predict',
        data: {
            'image_name': $("#file_preview").attr('src'),
            'age': parseInt($("#age").val()),
            'sex': $("#sex").val()
        },
        success: function(response) {
            $(".large-result").css("display", "block");
            $("#prediction_prob").text(response);
            console.log(response);
        }  
    })
})

$("#image_mole").change(function() {
    var file_item = document.getElementById('image_mole').files[0];
    var file_reader = new FileReader();
    file_reader.addEventListener('load', function() {
        document.getElementById('file_preview').src = file_reader.result
        console.log(file_reader.result)
    }, false);
    file_reader.readAsDataURL(file_item)
})