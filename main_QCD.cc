#include "Pythia8/Pythia.h"
#include<fstream>
using namespace Pythia8;

int main(){
    Pythia pythia;
    //QCD Hard Process

    pythia.readString("HardQCD:all=on");
    pythia.readString("Beams:eCM= 13000"); //lhc energy specifier 13TeV
    pythia.init();

    std::ofstream file("QCD_data.csv");
    file<<"pt,eta,phi,mass,label\n";

    for (int ievent = 0; ievent <10000 ; ++ievent)
    {
        if (!pythia.next()) continue;
        for (int i = 0; i < pythia.event.size(); ++i)
        {
            if (pythia.event[i].isFinal()) {
                double pt = pythia.event[i].pT();
                double eta = pythia.event[i].eta();
                double phi = pythia.event[i].phi();
                double mass = pythia.event[i].m();
                file << pt << "," << eta << "," << phi << "," << mass << ",0\n";

            }
        }
        
    }
    file.close();
    pythia.stat();
    return 0;

}