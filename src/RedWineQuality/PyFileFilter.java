package RedWineQuality;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class PyFileFilter extends FileFilter {
	public String getDescription() { 
        return "Python Script (*.py)";
    }
 
    public boolean accept(File f) {
        if (f.isDirectory()) {
            return true;
        } else {
            return f.getName().toLowerCase().endsWith(".py");
        }
    }
}
