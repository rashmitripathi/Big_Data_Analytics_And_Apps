<%@page import="java.util.List"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
	pageEncoding="ISO-8859-1"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">
<script type="text/javascript" src="http://code.jquery.com/jquery-1.7.1.min.js"></script>	
<link href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.0/css/bootstrap.min.css" rel="stylesheet">

<%
    List<String> keys= (List<String>) request.getAttribute("summaryList");
 %>

<script>
<%
String qDescription = "Summary of the vedio is : ";
for (int i = 0; i < keys.size(); i++) {%>
     console.log(<%=keys.get(i)%>);
    <%qDescription=qDescription+" "+keys.get(i);
}%>
</script>
<script src="http://code.jquery.com/jquery-1.11.1.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.0/js/bootstrap.min.js"></script>	
<title>Summarized data</title>
</head>
<style type="text/css">
     h1 { color: #ff4411; font-size: 30px; font-family: 'Signika', sans-serif; padding-bottom: 10px; }
     body { background-color: lightblue; }
    </style>
<body>
<h1>Get Summary of Video</h1>
    <form id="searchForm" class="list" action="getVideoSummary" target="_self">
        <div id="searchApp-bar3" class="button-bar">
            <label class="item item-input" >
                <i class="icon ion-search placeholder-icon"></i>
                <input  type="search" placeholder="Enter video URL" name="path" style="width: 900px; height: 40px;border-radius:38px 38px 38px 38px;">
            </label>
            <div class="spacer" style="width: 36px; height: 36px;"></div>
            <button type="submit" id="searchApp-button11" style="border-radius:38px 38px 38px 38px;" class="button button-dark  button-block">Get Summary >></button>
        </div>
    </form>
    <div class="spacer" style="width: 300px; height: 18px;"></div>
    <div id="summary" >
        <textarea style="width:100%;height:80%" name="comment" id="summary" class="form-control" rows="30" columns="100"><%=qDescription%></textarea>
    </div>
    </body>
</html>