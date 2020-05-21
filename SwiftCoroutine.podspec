Pod::Spec.new do |s|
  s.name = 'SwiftCoroutine'
  s.version = '2.1.2'
  s.license = 'MIT'
  s.summary = 'Swift coroutines for iOS, macOS and Linux.'
  s.homepage = 'https://github.com/belozierov/SwiftCoroutine'
  s.authors = { 'Alex Belozierov' => 'belozierov@gmail.com' }
  s.source = { :git => 'https://github.com/belozierov/SwiftCoroutine.git', :tag => s.version }
  s.documentation_url = 'https://belozierov.github.io/SwiftCoroutine/'

  s.ios.deployment_target = '10.0'
  s.osx.deployment_target = '10.12'

  s.swift_version = '5.2'

  s.source_files = ['Sources/**/*.{h,c,swift}', 'Sources/SwiftCoroutine.h']
end