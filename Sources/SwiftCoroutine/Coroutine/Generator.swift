//
//  Generator.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 02.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

open class Generator<Element> {
    
    private enum State {
        case prepared, started, finished
    }
    
    public typealias Iterator = ((Element?) -> Void) -> Void
    
    private let iterator: Iterator
    private var state: State = .prepared
    private var _next: Element?
    
    private lazy var coroutine: Coroutine = .new(block: { [weak self] in
        self?.iterator {
            self?._next = $0
            self?.coroutine.suspend()
        }
        self?._next = nil
        self?.state = .finished
    })
    
    public init(iterator: @escaping Iterator) {
        self.iterator = iterator
    }
    
    deinit {
        if state != .finished { coroutine.free() }
    }
    
}

extension Generator: IteratorProtocol {
    
    open func next() -> Element? {
        switch state {
        case .prepared:
            state = .started
            coroutine.start()
        case .started:
            coroutine.resume()
        case .finished:
            break
        }
        return _next
    }
    
}
