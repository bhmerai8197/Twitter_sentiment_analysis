function SearchLoader()
{
    if (validate_empty('twitter_search'))
    {
        var twitter_form_data = new FormData();
        var twitter_search_value = document.getElementById('twitter_search').value;
        twitter_form_data.append("twitter_search",twitter_search_value);
        ajaxCall('search',twitter_form_data,'load');

    }
}
function parse(spec) {

  vg.parse.spec(spec, function(chart) { chart({el:"#vis"}).update(); });

}
function chart()
{
    parse("json");


}

function chart_1()
{

    parse("json_1");

}

