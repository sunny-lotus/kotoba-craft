"""An original quiet atelier waltz. Score and source hashes are preserved."""
import pathlib,json,wave,sys,subprocess,hashlib
import numpy as np
folder=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(folder.parents[1]/'audio-research/python-tools'))
import imageio_ffmpeg
sr=22050;quarter=60/72;period=40;n=sr*period
def read(file):
 with wave.open(str(file)) as w:
  assert w.getsampwidth()==2
  x=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float).reshape(-1,w.getnchannels()).mean(axis=1)/32768
  return x,w.getframerate()
samples={str(pitch):(*read(folder/'sources'/f'piano-{pitch}.wav'),pitch) for pitch in [48,60,72]}
samples['wood']=(*read(folder/'sources/marimba-c4.wav'),72)
chords=[(41,[57,60]),(48,[55,62]),(50,[57,65]),(46,[58,62]),(43,[58,62]),(48,[55,64]),(41,[57,60]),(48,[55,62]),
        (50,[57,65]),(45,[57,60]),(46,[58,62]),(41,[57,60]),(43,[58,62]),(48,[55,64]),(48,[58,64]),(41,[57,60])]
phrases=[[(0,69,1.2),(1.5,72,1)],[(0,74,1.8)],[(.5,72,1),(2,69,.75)],[(0,67,1.8)],
 [(0,65,1),(1.5,67,1)],[(.5,69,1.5)],[(0,72,1),(1.5,69,1)],[(0,67,1.8)],
 [(0,65,1),(1.5,69,1)],[(.5,67,1.5)],[(0,65,1),(1.5,62,1)],[(0,60,1.8)],
 [(0,62,1),(1.5,65,1)],[(0,67,1),(1.5,69,1)],[(.5,67,1.8)],[(0,65,1.8)]]
events=[]
def note(instrument,midi,beat,hold,gain,pan):events.append(dict(instrument=instrument,midi=midi,beat=beat,hold=hold,gain=gain,pan=pan))
for bar,((bass,chord),phrase) in enumerate(zip(chords,phrases)):
 for beat,midi,hold in phrase:note('piano',midi,bar*3+beat,hold,.36,-.05)
 note('piano',bass,bar*3,1.8,.18,0)
 for pitch in chord:note('piano',pitch,bar*3+1.1,1.35,.11,.08)
 if bar%2==1:note('wood',chord[1]+12,bar*3+2,.45,.055,.20)
score=dict(title='ことばのアトリエ',version=1,meter='3/4',bpm=72,bars=16,loopSeconds=period,key='F major / D minor',barGains=[.78,.88,1,.82,.85,1,1.12,.86,.72,.82,.94,.78,.84,1,1.08,.76],events=events)
(folder/'score.json').write_text(json.dumps(score,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
mix=np.zeros((n*3+sr*4,2))
for cycle in range(3):
 for e in events:
  bank=samples['wood'] if e['instrument']=='wood' else samples[str(min([48,60,72],key=lambda p:abs(p-e['midi'])))];x,rate,pitch=bank
  speed=2**((e['midi']-pitch)/12);duration=min(len(x)/rate/speed-.003,e['hold']*quarter+.35);count=round(duration*sr);t=np.arange(count)/sr
  y=np.interp(t*rate*speed,np.arange(len(x)),x)*e['gain'];env=np.minimum(1,t/.008)*np.minimum(1,(duration-t)/.3);y*=np.clip(env,0,1)
  pan=np.array([np.sqrt((1-e['pan'])/2),np.sqrt((1+e['pan'])/2)]);at=round((cycle*period+e['beat']*quarter)*sr);mix[at:at+count]+=y[:,None]*pan
dry=mix.copy()
for delay,level in [(.083,.10),(.179,.055),(.311,.025)]:
 d=round(delay*sr);mix[d:]+=dry[:-d,::-1]*level
mix=mix[n:n*2];position=np.arange(n)/sr/(quarter*3);bar=position.astype(int);fraction=position-bar;blend=(1-np.cos(np.pi*fraction))/2;g=np.array(score['barGains']);mix*=(g[bar]*(1-blend)+g[(bar+1)%16]*blend)[:,None]
mix*=min(.05/max(np.sqrt(np.mean(mix**2)),1e-9),.42/max(abs(mix).max(),1e-9))
with wave.open(str(folder/'atelier-v1.wav'),'wb') as w:
 w.setnchannels(2);w.setsampwidth(2);w.setframerate(sr);w.writeframes((mix*32767).astype('<i2').tobytes())
out=folder.parent/'common/audio';out.mkdir(parents=True,exist_ok=True);mp3=out/'kotoba-atelier-v1.mp3';ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg,'-v','error','-y','-i',str(folder/'atelier-v1.wav'),'-codec:a','libmp3lame','-b:a','96k',str(mp3)],check=True)
b=subprocess.run([ffmpeg,'-v','error','-i',str(mp3),'-f','f32le','-acodec','pcm_f32le','-'],check=True,stdout=subprocess.PIPE).stdout;pcm=np.frombuffer(b,dtype='<f4').reshape(-1,2)
metrics=dict(seconds=len(pcm)/sr,bytes=mp3.stat().st_size,peak=float(abs(pcm).max()),rms=float(np.sqrt(np.mean(pcm**2))),boundaryJump=float(abs(pcm[0]-pcm[-1]).max()),sha256=hashlib.sha256(mp3.read_bytes()).hexdigest())
assert metrics['seconds']==40 and metrics['peak']<.5 and metrics['boundaryJump']<.02
(folder/'verification.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
sources=[dict(file='sources/'+p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),library='VCSL' if 'marimba' in p.name else 'FreePats Upright Piano KW',license='CC0-1.0') for p in sorted((folder/'sources').glob('*.wav')) if p.name in ['piano-48.wav','piano-60.wav','piano-72.wav','marimba-c4.wav']]
(folder/'sources.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8');print(json.dumps(metrics))
