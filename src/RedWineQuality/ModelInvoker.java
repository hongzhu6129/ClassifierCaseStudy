package RedWineQuality;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

public class ModelInvoker { 
	BufferedReader reader; 
	BufferedWriter writer;
	
	public ModelInvoker(String scriptFileName) throws IOException{
		String Python = "C:\\Users\\Hong\\AppData\\Local\\Programs\\Python\\Python39\\python.exe ";
//		String pyScript = "C:\\Morphy\\RedWineQuality\\PyScripts" + File.separator + scriptFileName;
		Process proc = Runtime.getRuntime().exec(Python + scriptFileName);
        reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
        writer = new BufferedWriter(new OutputStreamWriter(proc.getOutputStream()));
	}
	
	public String invokeModel(String param) throws IOException {
        writer.write(param+"\n");
        writer.flush();
        String line = reader.readLine();
        return line;
	}
}
