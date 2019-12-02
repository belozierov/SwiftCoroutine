//
//  Generator.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 02.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

open class Generator<Element> {
    
    private enum State {
        case prepared, suspended, resumed
    }
    
    public typealias Iterator = ((Element?) -> Void) -> Void
    
    private let iterator: Iterator
    private var state: State = .prepared
    
    private var _next: Element? {
        didSet {
            state = .suspended
            coroutine.suspend()
        }
    }
    
    private lazy var coroutine: Coroutine = .new(block: {
        [weak self] in self?.iterate()
    })
    
    public init(iterator: @escaping Iterator) {
        self.iterator = iterator
    }
    
    private func iterate() {
        iterator { _next = $0 }
        _next = nil
    }
    
}

extension Generator: IteratorProtocol {
    
    open func next() -> Element? {
        switch state {
        case .prepared:
            coroutine.start()
        case .suspended:
            coroutine.resume()
        case .resumed:
            fatalError()
        }
        return _next
    }
    
}
