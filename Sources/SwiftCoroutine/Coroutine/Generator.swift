//
//  Generator.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 02.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

open class Generator<Element> {
    
    public typealias Iterator = ((Element?) -> Void) -> Void
    
    private enum State {
        case prepared, started, finished
    }
    
    private let iterator: Iterator
    private var state: State = .prepared
    private var _next: Element?
    private lazy var coroutine = Coroutine.newFromPool(dispatcher: .sync)
    
    public init(iterator: @escaping Iterator) {
        self.iterator = iterator
    }
    
    private func start() {
        state = .started
        coroutine.start { [weak self] in
            self?.iterator {
                self?._next = $0
                self?.coroutine.suspend()
            }
            self?._next = nil
            self?.state = .finished
        }
    }
    
}

extension Generator: IteratorProtocol {
    
    open func next() -> Element? {
        switch state {
        case .prepared: start()
        case .started: coroutine.resume()
        case .finished: break
        }
        return _next
    }
    
}
